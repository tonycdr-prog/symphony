defmodule SymphonyElixir.AgentRunner do
  @moduledoc """
  Executes a single Linear issue in its workspace with Codex.
  """

  require Logger
  alias SymphonyElixir.Codex.AppServer
  alias SymphonyElixir.{Config, Linear.Issue, PromptBuilder, Tracker, Workspace}

  @type worker_host :: String.t() | nil

  @spec run(map(), pid() | nil, keyword()) :: :ok | no_return()
  def run(issue, codex_update_recipient \\ nil, opts \\ []) do
    # The orchestrator owns host retries so one worker lifetime never hops machines.
    worker_host = selected_worker_host(Keyword.get(opts, :worker_host), Config.settings!().worker.ssh_hosts)

    Logger.info("Starting agent run for #{issue_context(issue)} worker_host=#{worker_host_for_log(worker_host)}")

    case run_on_worker_host(issue, codex_update_recipient, opts, worker_host) do
      :ok ->
        :ok

      {:error, reason} ->
        Logger.error("Agent run failed for #{issue_context(issue)}: #{inspect(reason)}")
        raise RuntimeError, "Agent run failed for #{issue_context(issue)}: #{inspect(reason)}"
    end
  end

  defp run_on_worker_host(issue, codex_update_recipient, opts, worker_host) do
    Logger.info("Starting worker attempt for #{issue_context(issue)} worker_host=#{worker_host_for_log(worker_host)}")

    case Workspace.create_for_issue(issue, worker_host) do
      {:ok, workspace} ->
        send_worker_runtime_info(codex_update_recipient, issue, worker_host, workspace)

        try do
          with :ok <- Workspace.run_before_run_hook(workspace, issue, worker_host) do
            run_codex_turns(workspace, issue, codex_update_recipient, opts, worker_host)
          end
        after
          Workspace.run_after_run_hook(workspace, issue, worker_host)
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp codex_message_handler(recipient, issue, budget_tracker) do
    fn message ->
      update_token_budget(budget_tracker, message)
      send_codex_update(recipient, issue, message)
    end
  end

  defp send_codex_update(recipient, %Issue{id: issue_id}, message)
       when is_binary(issue_id) and is_pid(recipient) do
    send(recipient, {:codex_worker_update, issue_id, message})
    :ok
  end

  defp send_codex_update(_recipient, _issue, _message), do: :ok

  defp send_worker_runtime_info(recipient, %Issue{id: issue_id}, worker_host, workspace)
       when is_binary(issue_id) and is_pid(recipient) and is_binary(workspace) do
    send(
      recipient,
      {:worker_runtime_info, issue_id,
       %{
         worker_host: worker_host,
         workspace_path: workspace
       }}
    )

    :ok
  end

  defp send_worker_runtime_info(_recipient, _issue, _worker_host, _workspace), do: :ok

  defp run_codex_turns(workspace, issue, codex_update_recipient, opts, worker_host) do
    max_turns = Keyword.get(opts, :max_turns, Config.settings!().agent.max_turns)
    issue_state_fetcher = Keyword.get(opts, :issue_state_fetcher, &Tracker.fetch_issue_states_by_ids/1)

    with {:ok, session} <- AppServer.start_session(workspace, worker_host: worker_host) do
      case start_token_budget_tracker() do
        {:ok, budget_tracker} ->
          run_context = %{
            app_session: session,
            workspace: workspace,
            codex_update_recipient: codex_update_recipient,
            opts: opts,
            issue_state_fetcher: issue_state_fetcher,
            budget_tracker: budget_tracker,
            max_turns: max_turns
          }

          try do
            do_run_codex_turns(run_context, issue, 1)
          after
            AppServer.stop_session(session)
            stop_token_budget_tracker(budget_tracker)
          end

        {:error, reason} ->
          AppServer.stop_session(session)
          {:error, reason}
      end
    end
  end

  defp do_run_codex_turns(%{max_turns: max_turns} = context, issue, turn_number) do
    prompt = build_turn_prompt(issue, context.opts, turn_number, max_turns)

    with {:ok, turn_session} <-
           AppServer.run_turn(
             context.app_session,
             prompt,
             issue,
             on_message: codex_message_handler(context.codex_update_recipient, issue, context.budget_tracker)
           ) do
      Logger.info("Completed agent run for #{issue_context(issue)} session_id=#{turn_session[:session_id]} workspace=#{context.workspace} turn=#{turn_number}/#{max_turns}")

      case continue_with_issue?(issue, context.issue_state_fetcher) do
        {:continue, refreshed_issue} when turn_number < max_turns ->
          continue_or_halt_for_budget(context, refreshed_issue, turn_session, turn_number)

        {:continue, refreshed_issue} ->
          Logger.info("Reached agent.max_turns for #{issue_context(refreshed_issue)} with issue still active; returning control to orchestrator")

          :ok

        {:done, _refreshed_issue} ->
          :ok

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp continue_or_halt_for_budget(context, refreshed_issue, turn_session, turn_number) do
    case continuation_budget_violation(context.budget_tracker) do
      nil ->
        Logger.info("Continuing agent run for #{issue_context(refreshed_issue)} after normal turn completion turn=#{turn_number}/#{context.max_turns}")

        do_run_codex_turns(context, refreshed_issue, turn_number + 1)

      reason ->
        halt_for_continuation_budget(refreshed_issue, turn_session, reason)
    end
  end

  defp build_turn_prompt(issue, opts, 1, _max_turns), do: PromptBuilder.build_prompt(issue, opts)

  defp build_turn_prompt(_issue, _opts, turn_number, max_turns) do
    remaining_budget = continuation_budget_prompt_line()

    """
    Continuation guidance:

    - The previous Codex turn completed normally, but the Linear issue is still in an active state.
    - This is continuation turn ##{turn_number} of #{max_turns} for the current agent run.
    - Resume from the current workspace and workpad state instead of restarting from scratch.
    - The original task instructions and prior turn context are already present in this thread, so do not restate them before acting.
    #{remaining_budget}
    - Focus on the remaining ticket work and do not end the turn while the issue stays active unless you are truly blocked.
    """
  end

  defp start_token_budget_tracker do
    Agent.start_link(fn ->
      %{input_tokens: 0, output_tokens: 0, total_tokens: 0}
    end)
  end

  defp stop_token_budget_tracker(pid) when is_pid(pid), do: Agent.stop(pid, :normal)

  defp update_token_budget(pid, message) when is_pid(pid) and is_map(message) do
    case extract_token_usage(message) do
      %{total_tokens: total_tokens} = usage when is_integer(total_tokens) ->
        Agent.update(pid, fn totals ->
          %{
            input_tokens: max(totals.input_tokens, Map.get(usage, :input_tokens, 0)),
            output_tokens: max(totals.output_tokens, Map.get(usage, :output_tokens, 0)),
            total_tokens: max(totals.total_tokens, total_tokens)
          }
        end)

      _ ->
        :ok
    end
  end

  defp continuation_budget_violation(pid) when is_pid(pid) do
    settings = Config.settings!().codex
    max_total = settings.max_session_total_tokens

    if positive_integer?(max_total) do
      threshold = floor(max_total * settings.continuation_token_budget_ratio)
      totals = Agent.get(pid, & &1)

      if totals.total_tokens >= threshold do
        remaining = max(max_total - totals.total_tokens, 0)

        "codex.continuation_token_budget_ratio reached: #{totals.total_tokens}/#{max_total} tokens used, #{remaining} remaining, threshold=#{threshold}"
      end
    end
  end

  defp halt_for_continuation_budget(issue, turn_session, reason) do
    block_state = Config.settings!().tracker.block_state || "Todo"

    Logger.warning("Stopping continuation for #{issue_context(issue)} session_id=#{turn_session[:session_id]}: #{reason}; moving to #{inspect(block_state)}")

    case Tracker.update_issue_state(issue.id, block_state) do
      :ok ->
        :ok

      {:error, update_reason} ->
        Logger.error("Could not move #{issue_context(issue)} to #{inspect(block_state)} after budget stop: #{inspect(update_reason)}")
    end

    :ok
  end

  defp continuation_budget_prompt_line do
    settings = Config.settings!().codex

    if positive_integer?(settings.max_session_total_tokens) do
      threshold = floor(settings.max_session_total_tokens * settings.continuation_token_budget_ratio)

      "- Keep this continuation under the session budget. If the remaining work would require broad file reads or exploration, update the workpad with a blocker instead of continuing. Continuations stop automatically at #{threshold}/#{settings.max_session_total_tokens} tracked tokens."
    else
      "- Keep this continuation bounded. If the remaining work would require broad file reads or exploration, update the workpad with a blocker instead of continuing."
    end
  end

  defp extract_token_usage(message) when is_map(message) do
    payloads = [
      message[:usage],
      Map.get(message, "usage"),
      message[:payload],
      Map.get(message, "payload"),
      message
    ]

    Enum.find_value(payloads, &token_usage_from_payload/1)
  end

  defp token_usage_from_payload(payload) when is_map(payload) do
    cond do
      token_usage_map?(payload) ->
        normalize_token_usage(payload)

      usage = map_at_path(payload, ["params", "tokenUsage", "total"]) ->
        normalize_token_usage(usage)

      usage = map_at_path(payload, [:params, :tokenUsage, :total]) ->
        normalize_token_usage(usage)

      usage = map_at_path(payload, ["params", "msg", "payload", "info", "total_token_usage"]) ->
        normalize_token_usage(usage)

      usage = map_at_path(payload, [:params, :msg, :payload, :info, :total_token_usage]) ->
        normalize_token_usage(usage)

      true ->
        nil
    end
  end

  defp token_usage_from_payload(_payload), do: nil

  defp token_usage_map?(payload) when is_map(payload) do
    token_integer(payload, ["total_tokens", :total_tokens, "totalTokens", :totalTokens]) != nil
  end

  defp normalize_token_usage(payload) when is_map(payload) do
    %{
      input_tokens: token_integer(payload, ["input_tokens", :input_tokens, "inputTokens", :inputTokens]) || 0,
      output_tokens: token_integer(payload, ["output_tokens", :output_tokens, "outputTokens", :outputTokens]) || 0,
      total_tokens: token_integer(payload, ["total_tokens", :total_tokens, "totalTokens", :totalTokens]) || 0
    }
  end

  defp map_at_path(payload, path) when is_map(payload) and is_list(path) do
    Enum.reduce_while(path, payload, fn key, acc ->
      case acc do
        %{} -> {:cont, Map.get(acc, key)}
        _ -> {:halt, nil}
      end
    end)
  end

  defp token_integer(payload, keys) when is_map(payload) and is_list(keys) do
    Enum.find_value(keys, fn key ->
      case Map.get(payload, key) do
        value when is_integer(value) -> value
        _ -> nil
      end
    end)
  end

  defp continue_with_issue?(%Issue{id: issue_id} = issue, issue_state_fetcher) when is_binary(issue_id) do
    case issue_state_fetcher.([issue_id]) do
      {:ok, [%Issue{} = refreshed_issue | _]} ->
        if active_issue_state?(refreshed_issue.state) do
          {:continue, refreshed_issue}
        else
          {:done, refreshed_issue}
        end

      {:ok, []} ->
        {:done, issue}

      {:error, reason} ->
        {:error, {:issue_state_refresh_failed, reason}}
    end
  end

  defp continue_with_issue?(issue, _issue_state_fetcher), do: {:done, issue}

  defp active_issue_state?(state_name) when is_binary(state_name) do
    normalized_state = normalize_issue_state(state_name)

    Config.settings!().tracker.active_states
    |> Enum.any?(fn active_state -> normalize_issue_state(active_state) == normalized_state end)
  end

  defp active_issue_state?(_state_name), do: false

  defp selected_worker_host(nil, []), do: nil

  defp selected_worker_host(preferred_host, configured_hosts) when is_list(configured_hosts) do
    hosts =
      configured_hosts
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))
      |> Enum.uniq()

    case preferred_host do
      host when is_binary(host) and host != "" -> host
      _ when hosts == [] -> nil
      _ -> List.first(hosts)
    end
  end

  defp worker_host_for_log(nil), do: "local"
  defp worker_host_for_log(worker_host), do: worker_host

  defp normalize_issue_state(state_name) when is_binary(state_name) do
    state_name
    |> String.trim()
    |> String.downcase()
  end

  defp positive_integer?(value), do: is_integer(value) and value > 0

  defp issue_context(%Issue{id: issue_id, identifier: identifier}) do
    "issue_id=#{issue_id} issue_identifier=#{identifier}"
  end
end
