// The Q&A page (spec P3-04, design s16-9). It asks `POST /v1/ask_docs/stream` and reads the
// response with fetch: EventSource cannot send a POST. Events (see rag/serve/stream.py):
//   progress             a node started or ended: the status line says what the server is doing
//   token                a piece of the answer as the model writes it: shown as plain text
//   answer               the checked answer: replaces the streamed text; markers become buttons
//   need_clarification   the server asks back; the box below sends the reply with the resume token
//   error                the category and the request id; done  numbers of the run
// Pressing Cancel closes the connection (AbortController): the server then stops the run.
(function () {
  "use strict";

  const R = window.RagRender;
  const { SseParser } = window.RagSse;
  const $ = (id) => document.getElementById(id);
  const ui = {
    kb: $("kb"),
    question: $("question"),
    ask: $("ask"),
    cancel: $("cancel"),
    newChat: $("new-chat"),
    status: $("status"),
    live: $("live"),
    answer: $("answer"),
    removed: $("removed"),
    notice: $("notice"),
    sources: $("sources"),
    passage: $("passage"),
    clarify: $("clarify"),
    clarifyQuestion: $("clarify-question"),
    clarifyReply: $("clarify-reply"),
    clarifySend: $("clarify-send"),
    error: $("error"),
    feedback: $("feedback"),
    useful: $("useful"),
    notUseful: $("not-useful"),
    feedbackStatus: $("feedback-status"),
  };

  const state = {
    controller: null,
    running: false,
    attempt: 1,
    request: null, // {kb, question}: what the last call asked, for a clarification reply
    resumeToken: null,
    requestId: null,
    rated: false,
  };

  // ---- the conversation id lives as long as the browser tab ---------------------------------

  const memory = {};
  function stored(key) {
    try {
      return window.sessionStorage.getItem(key);
    } catch (e) {
      return memory[key] || null;
    }
  }
  function store(key, value) {
    try {
      window.sessionStorage.setItem(key, value);
    } catch (e) {
      memory[key] = value;
    }
  }
  function newSessionId() {
    const random =
      window.crypto && window.crypto.randomUUID
        ? window.crypto.randomUUID()
        : Array.from({ length: 4 }, () => Math.random().toString(16).slice(2, 10)).join("");
    return "web-" + random.replace(/[^A-Za-z0-9._-]/g, "").slice(0, 40);
  }
  function sessionId() {
    let id = stored("rag.session");
    if (!id) {
      id = newSessionId();
      store("rag.session", id);
    }
    return id;
  }

  // ---- small helpers ------------------------------------------------------------------------

  function setStatus(text) {
    ui.status.textContent = text;
  }
  function setRunning(running) {
    state.running = running;
    ui.ask.disabled = running;
    ui.cancel.hidden = !running;
    ui.kb.disabled = running;
    ui.clarifySend.disabled = running;
  }
  // `keepClarification`: a reply to a question asked back is being sent; the question, the
  // reply and the resume token stay until that call has ended with an answer, so a reply
  // that fails or is cancelled can be sent again (the server gives the token back).
  function resetOutput(keepClarification) {
    for (const node of [ui.live, ui.answer, ui.removed, ui.notice, ui.sources, ui.passage, ui.error]) {
      R.clear(node);
      node.hidden = true;
    }
    ui.live.className = "live";
    if (!keepClarification) dropClarification();
    ui.feedback.hidden = true;
    ui.feedbackStatus.textContent = "";
    ui.useful.disabled = false;
    ui.notUseful.disabled = false;
    state.attempt = 1;
    state.rated = false;
    state.requestId = null;
  }
  function dropClarification() {
    ui.clarify.hidden = true;
    ui.clarifyReply.value = "";
    state.resumeToken = null;
  }
  // Text that was streamed before the run failed: it was not checked against the sources.
  function markUnverified() {
    if (!ui.live.textContent) return;
    ui.live.hidden = false;
    ui.live.className = "live unverified";
    ui.notice.textContent =
      "The grey text was written before the run stopped. It was not checked against the " +
      "sources and may be incomplete or wrong.";
    ui.notice.hidden = false;
  }
  function showError(error) {
    if (error.error === "invalid_resume_token") dropClarification(); // used up or expired
    R.renderError(document, ui.error, error);
    ui.error.hidden = false;
  }
  function showPassage(source, marker) {
    R.renderPassage(document, ui.passage, source, marker);
    ui.passage.hidden = false;
  }

  // ---- what the server says it is doing -----------------------------------------------------

  const DOING = {
    route: "Reading the question…",
    clarify: "Checking whether the question is clear…",
    rewrite: "Rewriting the question for search…",
    retrieve: "Searching the documents…",
    judge: "Checking whether what was found is enough…",
    answer: "Writing the answer…",
    verify: "Checking the citations…",
  };
  function describe(event) {
    if (event.node === "queued") {
      return "Waiting: an earlier question of this conversation is still running.";
    }
    if (event.node === "answer" && event.phase === "retry") {
      return "The first attempt cited nothing; writing the answer again…";
    }
    const round = event.round > 1 ? " (round " + event.round + ")" : "";
    if (event.phase === "end" && event.node === "retrieve") {
      return "Found " + event.chunks + " passage" + (event.chunks === 1 ? "" : "s") + round + ".";
    }
    return (DOING[event.node] || "Working…") + (event.node === "retrieve" ? round : "");
  }

  // ---- events -------------------------------------------------------------------------------

  function onProgress(event) {
    if (event.node === "answer" && event.phase === "retry") {
      R.clear(ui.live); // the text so far is void
      state.attempt = event.attempt;
    }
    setStatus(describe(event));
  }

  function onToken(event) {
    if (event.attempt !== state.attempt) {
      R.clear(ui.live);
      state.attempt = event.attempt;
    }
    ui.live.hidden = false;
    ui.live.className = "live";
    R.appendText(document, ui.live, event.text);
  }

  function onAnswer(response) {
    state.requestId = response.request_id;
    R.clear(ui.live);
    ui.live.hidden = true;
    dropClarification(); // the run ended with an answer
    R.renderAnswer(document, ui.answer, response, (marker, source) => showPassage(source, marker));
    ui.answer.hidden = false;
    R.renderRemoved(document, ui.removed, response.citation_check && response.citation_check.removed_markers);
    ui.removed.hidden = !(response.citation_check && response.citation_check.removed_markers.length);
    const notes = [];
    if (response.notice) notes.push(response.notice);
    if (response.degraded) notes.push("Only part of the search chain answered, so results may be incomplete.");
    if (response.status === "not_found" && response.sentences.length === 0) {
      notes.push("No answer could be written from what was found.");
    }
    if (notes.length) {
      ui.notice.textContent = notes.join(" ");
      ui.notice.hidden = false;
    }
    const sources = response.sources && response.sources.length ? response.sources : response.nearest;
    if (sources && sources.length) {
      const heading = response.sources && response.sources.length ? "Sources" : "Closest passages (not an answer)";
      ui.sources.hidden = false;
      R.renderSourceList(document, ui.sources, sources, (source, n) => showPassage(source, n));
      const title = document.createElement("p");
      title.className = "list-title";
      title.appendChild(document.createTextNode(heading));
      ui.sources.insertBefore(title, ui.sources.firstChild);
    }
  }

  function onClarification(event) {
    state.resumeToken = event.resume_token;
    state.requestId = event.request_id;
    ui.clarifyQuestion.textContent = event.question;
    ui.clarifyReply.value = "";
    ui.clarify.hidden = false;
    ui.clarifyReply.focus();
  }

  function onDone(event) {
    const t = event.timings_ms || {};
    const seconds = (t.request / 1000).toFixed(1);
    const first = t.first_progress !== undefined ? " First progress after " + Math.round(t.first_progress) + " ms" : "";
    const text = t.first_token !== undefined ? ", first text after " + Math.round(t.first_token) + " ms." : ".";
    if (event.status === "need_clarification") {
      setStatus("The question needs one more detail.");
    } else {
      setStatus("Done in " + seconds + " s." + first + text);
      ui.feedback.hidden = false;
    }
  }

  function handle(name, text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      return false; // not one of ours: skipped
    }
    switch (name) {
      case "progress":
        onProgress(data);
        break;
      case "token":
        onToken(data);
        break;
      case "answer":
        onAnswer(data);
        break;
      case "need_clarification":
        onClarification(data);
        break;
      case "error":
        showError(data);
        markUnverified();
        setStatus("Stopped by an error.");
        return true;
      case "done":
        onDone(data);
        return true;
      default:
        break; // an event this page does not know: ignored
    }
    return false;
  }

  // ---- one call -----------------------------------------------------------------------------

  async function run(body, resume) {
    resetOutput(resume);
    setRunning(true);
    state.controller = new AbortController();
    setStatus("Sending the question…");
    let finished = false;
    try {
      const response = await fetch("/v1/ask_docs/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: state.controller.signal,
      });
      if (!response.ok) {
        let error = { error: "http_" + response.status, request_id: response.headers.get("X-Request-Id") };
        try {
          error = await response.json();
        } catch (e) {
          // not JSON: keep the status
        }
        showError(error);
        setStatus("The server did not accept the question.");
        return;
      }
      state.requestId = response.headers.get("X-Request-Id");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      const parser = new SseParser();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        for (const event of parser.push(decoder.decode(value, { stream: true }))) {
          if (handle(event.event, event.data)) finished = true;
        }
      }
      if (!finished) {
        markUnverified();
        setStatus("The connection ended before the answer was complete.");
      }
    } catch (error) {
      if (error && error.name === "AbortError") {
        R.clear(ui.live);
        ui.live.hidden = true;
        setStatus("Cancelled. The server stops working on it.");
      } else {
        showError({ error: "connection_lost", message: "The connection to the server was lost." });
        markUnverified();
        setStatus("Stopped by an error.");
      }
    } finally {
      setRunning(false);
      state.controller = null;
    }
  }

  function ask() {
    const question = ui.question.value.trim();
    if (!question || state.running) return;
    const kb = ui.kb.value;
    if (!kb) {
      setStatus("Choose a knowledge base first.");
      return;
    }
    state.request = { kb, question };
    run({ kb, question, session_id: sessionId() });
  }

  function sendClarification() {
    const reply = ui.clarifyReply.value.trim();
    if (!reply || state.running || !state.resumeToken || !state.request) return;
    run(
      {
        kb: state.request.kb,
        question: state.request.question,
        session_id: sessionId(),
        resume_token: state.resumeToken,
        answer: reply,
      },
      true
    );
  }

  async function sendFeedback(useful) {
    if (state.rated || !state.requestId) return;
    state.rated = true;
    ui.useful.disabled = true;
    ui.notUseful.disabled = true;
    try {
      const response = await fetch("/v1/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: state.requestId, useful }),
      });
      ui.feedbackStatus.textContent = response.ok ? "Thank you. Your answer was recorded." : "The feedback was not recorded.";
    } catch (e) {
      ui.feedbackStatus.textContent = "The feedback was not recorded.";
    }
  }

  // ---- start --------------------------------------------------------------------------------

  async function loadKnowledgeBases() {
    try {
      const response = await fetch("/v1/knowledge_bases");
      if (!response.ok) throw new Error("status " + response.status);
      const body = await response.json();
      const wanted = stored("rag.kb");
      for (const info of body.knowledge_bases) {
        const option = document.createElement("option");
        option.value = info.kb;
        option.appendChild(document.createTextNode(info.kb));
        option.title = info.description || "";
        option.disabled = info.available === false;
        ui.kb.appendChild(option);
      }
      if (wanted && Array.from(ui.kb.options).some((o) => o.value === wanted && !o.disabled)) {
        ui.kb.value = wanted;
      }
      if (body.knowledge_bases.length === 0) setStatus("No knowledge base is available to you.");
    } catch (e) {
      setStatus("The list of knowledge bases could not be loaded.");
    }
  }

  ui.ask.addEventListener("click", ask);
  ui.cancel.addEventListener("click", () => state.controller && state.controller.abort());
  ui.clarifySend.addEventListener("click", sendClarification);
  ui.useful.addEventListener("click", () => sendFeedback(true));
  ui.notUseful.addEventListener("click", () => sendFeedback(false));
  ui.kb.addEventListener("change", () => store("rag.kb", ui.kb.value));
  ui.newChat.addEventListener("click", () => {
    store("rag.session", newSessionId());
    resetOutput();
    setStatus("New conversation. Earlier questions are forgotten.");
  });
  ui.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      ask();
    }
  });
  ui.clarifyReply.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendClarification();
    }
  });

  setRunning(false);
  resetOutput();
  loadKnowledgeBases();
})();
