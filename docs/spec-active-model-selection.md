# Spec: Check the model as soon as it's picked

Status: Implemented
Files: `agentseed/server.py`, `agentseed/providers.py`, `agentseed/index.html`

## 1. Problem

Picking a provider/model in the dropdown only saves it in the browser
(`localStorage["agentseedModel"]`). The server hears about it only on the next
`POST /agent/chat?service=&model=`. As a result:

- A bad pick (missing API key, expired AWS SSO login, no access to the model)
  shows up only after the user has typed and sent a message. The turn is then
  rolled back and an error is shown.
- The first message to a newly picked model is slow, because the server builds
  that model's agent (`build_agent`) during the chat request.
- `/config` sets `live` from a check of each provider's **default** model only.
  Every other model from that provider gets the same status, which can be
  wrong.

## 2. Goal

As soon as the user picks a model in the dropdown, the server should:

1. check the pair is listed in `models.json`,
2. build the agent and keep it for later requests,
3. send that **exact** model a tiny test request ("ping"),
4. report back whether it works, before the user can send a message.

The UI shows the result straight away. If the model doesn't work, the UI goes
back to the last working pick.

## 3. Non-goals

- The server still doesn't remember which model each browser picked. The
  `service`/`model` query params on `/agent/chat` stay the only thing that
  decides which model answers. Picking a model must **not** change the server
  defaults (the first chat model in `models.json`, `app.state.agent`), because the server is shared by
  every open tab.
- No streaming, and no change to the chat request or response format.
- No checking of every model when the page loads. Only the picked model is
  checked (decision 2).
- The client never writes the "model switched" note. The server adds it at
  run time, and it is never sent again on later turns (4.5).

## 4. Backend

### 4.1 State

| Field | Type | Purpose |
| --- | --- | --- |
| `app.state.agents` | `dict[tuple[str, str], Agent]` | Already exists. Agents already built, keyed by `(service, model)`. |
| `app.state.model_status` | `dict[tuple[str, str], dict]` | **New.** Latest check result for each `(service, model)`. |
| `app.state.activation_tasks` | `dict[tuple[str, str], asyncio.Task]` | **New.** Checks still running, so two requests for the same pair share one check. |

Status record, which adds a timestamp to what `check_model` returns:

```json
{
  "service": "anthropic",
  "model": "claude-haiku-4-5",
  "live": true,
  "latency_ms": 412,
  "error": null,
  "checked_at": "2025-01-01T12:00:00Z"
}
```

When the startup check finishes (`probe_providers`), it also writes into
`model_status`, keyed by each provider's default model. That way the startup
results and the new on-pick results live in one place.

### 4.2 New endpoint: `POST /agent/activate`

Query params: `service` (required), `model` (optional; defaults to
`default_model(service)`), `force` (optional bool, default `false`).

Steps:

1. Look up the pair in `chat_model_options()`. If it isn't there, return
   **400** `{"detail": "unknown provider/model: X:Y"}`.
2. If `force` is false, there is a saved status younger than
   `ACTIVATE_TTL_S` (default 300s, set by environment variable), and that
   status says `live: true`, return the saved status.
3. If a check for this pair is already running, wait for it and return its
   result.
4. Otherwise start a check task that:
   1. builds the agent with `await asyncio.to_thread(build_agent, service, model)`
      (on Bedrock, building talks to AWS, so it must not block the server), and
      stores it in `app.state.agents[key]`;
   2. runs `check_model(service, model)`;
   3. saves the result in `model_status[key]` with `checked_at`;
   4. on failure, removes `agents[key]` so a broken agent isn't reused.
5. Return **200** with the status record. A model that doesn't work is still a
   successful *check*: the response has `live: false` and `error` set. 4xx/5xx
   are only for bad input or bugs in the server.

Failed results are saved too, but step 2 ignores them, so picking the same
model again checks it again. This covers a user who fixes the problem (for
example by running `aws sso login`) and picks the model again.

### 4.3 Changes to existing code

- **`require_agent`**: no change in how it works. It finds the agent that
  `/agent/activate` already built in `app.state.agents`, so the first chat
  message after a pick isn't slow. Building is moved off the server's main
  loop, same as 4.2.4.1.
- **`/config`**: each model's `live`/`error` comes from
  `model_status[(service, model)]`. If that model hasn't been checked, fall
  back to the provider's default-model status, and set a new field
  `checked: false`. The fallback is a guess and not a result for that model.
  Also include `latency_ms` when available.
- **`agent_chat`**: after a successful `agent.run`, set
  `model_status[key].live = true`. On an exception from the provider, set
  `live = false` with the error message. This keeps `/config` accurate without
  extra test requests.

### 4.4 Concurrency and cost

- Each test request costs about 16 output tokens. The TTL plus sharing a
  running check limits how often they're sent when users click through the
  dropdown quickly.
- A check gives up after `check_model`'s existing `timeout` (20s).
- At shutdown, `lifespan` cancels any checks still running in
  `activation_tasks`.

### 4.5 Telling the new model it took over (decision 1)

When a conversation switches models, the new model should know that earlier
assistant replies came from a different model. Otherwise it may defend or
repeat things it never said.

**Recording which model wrote each reply.** Before returning, `agent_chat`
labels each *new* `ModelResponse` (index `>= len(history)`) with
`metadata["agentseed_model"] = "service:model"`. It then sends the history
back with `ModelMessagesTypeAdapter.dump_json(messages)` instead of
`result.all_messages_json()`.

**Spotting a switch.** A helper `previous_model(history) -> str | None` returns
the label on the last `ModelResponse`. If there's no label (for example,
history saved before this change), it falls back to
`f"{provider_name}:{model_name}"`. Messages that aren't `ModelResponse`
are skipped. If no reply has a model name, it returns None.

**Telling the model.** If `previous_model(history)` is set and isn't the
current `service:model`, `agent_chat` runs:

```python
agent.run(
    message_history=history,
    instructions=SWITCH_NOTE.format(previous=prev, current=f"{service}:{model}"),
)
```

with

```python
SWITCH_NOTE = (
    "Earlier assistant turns in this conversation were written by {previous}. "
    "You are {current} and are taking over from here. Treat those turns as "
    "context, not as your own statements; correct them if they are wrong."
)
```

Notes:

- `instructions` passed to `run()` are added on top of the agent's
  `INSTRUCTIONS`. They don't replace them.
- Pydantic AI saves the instructions used for a turn on that turn's
  `ModelRequest.instructions`, so the note shows up in the returned history.
  It only sends the *current* run's instructions to the model, so a saved
  note is never sent again. On the next turn, the last reply is labelled with
  the current model, so no note is added. This is intended.
- The labels in `metadata` come from the client and could be faked. The only
  effect of a fake label is a wrong or missing note, so they don't need to be
  checked. `sanitize_messages` still removes system prompts sent by the
  client, as before.
- Defaults are filled in by one function, `resolve_chat_model(service, model)`
  in `providers.py`. No service means the first provider in `models.json`; no
  model means that provider's first model. It also rejects pairs that aren't
  listed. `require_agent` uses it (through `requested_pair`, which turns the
  error into a 400), so `agent_chat` gets the same `service:model` label.

### 4.6 Size limit on cached agents (decision 3)

Change `app.state.agents` from a plain `dict` to a least-recently-used cache
(`collections.OrderedDict`) with at most `AGENT_CACHE_SIZE` entries (default
8, set by environment variable).

- Getting an agent moves it to the most-recent end (`move_to_end(key)`).
- Adding an agent when the cache is full drops the oldest one
  (`popitem(last=False)`) and logs `Evicted agent 'service:model'`.
- The startup agent (`app.state.agent`) is kept outside this cache and is
  never dropped.
- Dropping an agent doesn't affect its `model_status` entry. Status records are
  small and stay. If a dropped model is picked again, the agent is rebuilt.
  The check is skipped if its status is still within the TTL.
- Put this logic in a small helper (`AgentCache` with `get`/`put`/`pop`), so
  `require_agent` and `/agent/activate` share it.

## 5. Frontend (`index.html`)

### 5.1 State

```js
const activation = ref({ key: "", state: "idle", error: null, latencyMs: null });
// state: "idle" | "activating" | "ready" | "failed"
let lastGood = "";            // last selection that came back live
let activationSeq = 0;        // guards against out-of-order responses
```

### 5.2 Flow

```js
watch(selected, async (key, prev) => {
  localStorage.setItem(SELECTED_KEY, key);
  const option = config.value?.models.find((o) => optionKey(o) === key);
  if (!option) return;
  const seq = ++activationSeq;
  activation.value = { key, state: "activating", error: null, latencyMs: null };
  try {
    const s = await ky.post(`/agent/activate?${new URLSearchParams(option)}`,
                            { timeout: 30000 }).json();
    if (seq !== activationSeq) return;          // user picked again; drop
    updateOptionStatus(s);                       // patch config.models entry
    if (s.live) {
      lastGood = key;
      activation.value = { key, state: "ready", error: null, latencyMs: s.latency_ms };
    } else {
      activation.value = { key, state: "failed", error: s.error, latencyMs: null };
      if (lastGood && lastGood !== key) selected.value = lastGood;  // revert
    }
  } catch (e) {
    if (seq !== activationSeq) return;
    activation.value = { key, state: "failed", error: e.message, latencyMs: null };
  }
});
```

- Also run this check once on mount, after the first `fetchConfig`, so a pick
  saved in `localStorage` is checked on page load.
- `fixSelection` keeps its current role (moving off picks already known to be
  dead). It changes `selected`, which starts the check through the `watch`
  above.

### 5.3 UI

- Show a small badge next to the dropdown:
  - `activating`: spinner, "connecting…"
  - `ready`: green dot, with the latency as a tooltip, e.g. "412 ms"
  - `failed`: red dot, with the error as a tooltip. If the UI went back to the
    previous model, also show an error bubble: "Couldn't switch to X: `{error}`.
    Still using Y."
- Disable the **Send** button while `activation.state === "activating"`. Typing
  stays allowed.
- Options marked `checked: false` show no status label. Options known to be
  dead stay disabled, as now.

## 6. API summary

| Method | Path | Change |
| --- | --- | --- |
| `POST` | `/agent/activate?service=&model=&force=` | **New.** Builds the agent, checks the model, returns its status. |
| `GET` | `/config` | Status per model; adds `checked` and `latency_ms`. |
| `POST` | `/agent/chat?service=&model=` | Same format; updates `model_status`, labels new replies in `metadata.agentseed_model`, and adds the switch note when the model changed. |

## 7. Acceptance criteria

1. Picking a working model shows "ready" within one round trip. The next chat
   message doesn't build the agent again (check the logs: no second
   `Initializing agent on ...`).
2. Picking a model with a missing API key shows "failed" with the
   `require_env` message and goes back to the previous model. No chat turn is
   used up.
3. Clicking through five models quickly leaves the UI showing the status of
   the **last** one picked. Responses that arrive late are ignored.
4. Two tabs picking the same pair at the same moment cause one test request,
   not two.
5. The server defaults don't change: `/config` `service`/`model` and
   requests to `/agent/chat` with no query params still use the startup agent.
6. A non-default model that the account can't use shows as unavailable in
   `/config` once checked, even if the provider's default model works.
7. After switching from A to B, B's first reply follows the switch note: it
   doesn't claim A's earlier replies as its own. B's second reply runs with no
   switch note.
8. Every reply returned by `/agent/chat` has `metadata.agentseed_model`
   matching the model that wrote it.
9. With `AGENT_CACHE_SIZE=2`, activating three models drops the one used least
   recently. The startup agent still answers requests with no query params.

## 8. Decisions

1. **Tell the new model it took over: yes.** The server adds a note for that
   run only, based on the label on earlier replies (4.5).
2. **Check every model on page load: no.** Only the picked model is checked,
   so there's no wave of about 25 test requests.
3. **Size limit on cached agents: yes.** Least-recently-used cache,
   `AGENT_CACHE_SIZE` (default 8), startup agent never dropped (4.6).
