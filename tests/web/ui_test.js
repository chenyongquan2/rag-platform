// Tests of the page's plain JavaScript (spec P3-04 AC24, AC25), run by
// tests/unit/test_serve_web.py:  node tests/web/ui_test.js sse|render
// No framework and no real DOM: a small fake document records every element the code creates and
// throws when the code tries to set HTML, so a test can say "this text became text, not markup".
"use strict";

const path = require("path");
const web = path.join(__dirname, "..", "..", "src", "rag", "serve", "web");
const { SseParser } = require(path.join(web, "sse.js"));
const R = require(path.join(web, "render.js"));

const failures = [];
function check(condition, message) {
  if (!condition) failures.push(message);
}
function same(actual, expected, message) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) failures.push(`${message}: expected ${e}, got ${a}`);
}

// ---- SSE parser ----------------------------------------------------------------------------

function parseAll(chunks) {
  const parser = new SseParser();
  const out = [];
  for (const chunk of chunks) out.push(...parser.push(chunk));
  return out;
}

function suiteSse() {
  const stream =
    ': keep-alive\n\n' +
    'event: progress\ndata: {"node":"route","phase":"start"}\n\n' +
    'event: token\ndata: {"text":"Hel","attempt":1}\n\n' +
    ': keep-alive\n\n' +
    'event: done\ndata: {"status":"answered"}\n\n';
  const whole = parseAll([stream]);
  same(
    whole.map((e) => e.event),
    ["progress", "token", "done"],
    "events of a whole stream, comments skipped"
  );
  same(JSON.parse(whole[1].data), { text: "Hel", attempt: 1 }, "data of an event");

  // The same stream split at every possible place, and one character at a time.
  for (let cut = 1; cut < stream.length; cut++) {
    same(parseAll([stream.slice(0, cut), stream.slice(cut)]), whole, `split at ${cut}`);
  }
  same(parseAll(stream.split("")), whole, "one character at a time");

  same(
    parseAll(["data: a\ndata: b\ndata:\ndata: c\n\n"]),
    [{ event: "message", data: "a\nb\n\nc" }],
    "several data lines are joined with a newline; the default name is message"
  );
  same(
    parseAll(["event: x\r\ndata: 1\r\n\r\nevent: y\rdata: 2\r\r"]),
    [
      { event: "x", data: "1" },
      { event: "y", data: "2" },
    ],
    "CRLF and lone CR line ends"
  );
  same(
    parseAll(["event: x\r", "\ndata: 1\r", "\n\r", "\n"]),
    [{ event: "x", data: "1" }],
    "a CR and its LF in different chunks are one line end"
  );
  same(
    parseAll(["event:progress\ndata:{\"a\":1}\n\n", "data:  two spaces\n\n"]),
    [
      { event: "progress", data: '{"a":1}' },
      { event: "message", data: " two spaces" },
    ],
    "no space after the colon; only one leading space is dropped"
  );
  same(
    parseAll(["event: weird\ndata: x\n\nid: 7\nretry: 100\nevent: done\ndata: {}\n\n"]),
    [
      { event: "weird", data: "x" },
      { event: "done", data: "{}" },
    ],
    "unknown events pass through, id and retry fields are ignored"
  );
  same(parseAll(["event: lonely\n\nevent: done\ndata: {}\n\n"]).map((e) => e.event), ["done"],
    "an event with no data is not dispatched, and its name does not leak into the next one");
  same(parseAll(["event: cut\ndata: {\"a\":"]), [], "an unfinished event is not dispatched");
  same(parseAll(["event: a\ndata: ä\n\n"]), [{ event: "a", data: "ä" }], "non-ASCII text");

  // Bytes split inside a multi-byte character are the decoder's business: stream mode.
  const bytes = Buffer.from('event: t\ndata: {"text":"héllo 你好 😀"}\n\n', "utf8");
  const decoder = new TextDecoder("utf-8");
  const parser = new SseParser();
  const got = [];
  for (let i = 0; i < bytes.length; i++) {
    got.push(...parser.push(decoder.decode(bytes.subarray(i, i + 1), { stream: true })));
  }
  same(got.map((e) => JSON.parse(e.data).text), ["héllo 你好 😀"],
    "multi-byte characters split between chunks");

  // A huge event is cut off, not buffered without end.
  const small = new SseParser({ maxBytes: 1000 });
  let threw = false;
  try {
    small.push("data: " + "x".repeat(5000));
  } catch (e) {
    threw = true;
  }
  check(threw, "a line longer than the limit makes the parser throw");
}

// ---- the fake document --------------------------------------------------------------------

class FakeText {
  constructor(data) {
    this.nodeType = 3;
    this.data = String(data);
  }
  get textContent() {
    return this.data;
  }
}

function makeDocument() {
  const created = [];
  class FakeElement {
    constructor(tag) {
      this.nodeType = 1;
      this.tag = tag;
      this.children = [];
      this.attrs = {};
      this.listeners = {};
      this.classes = new Set();
      this.dataset = {};
      this.hidden = false;
      created.push(this);
    }
    get classList() {
      const set = this.classes;
      return {
        add: (...names) => names.forEach((n) => set.add(n)),
        remove: (...names) => names.forEach((n) => set.delete(n)),
      };
    }
    set className(value) {
      this.classes = new Set(String(value).split(/\s+/).filter(Boolean));
    }
    get className() {
      return [...this.classes].join(" ");
    }
    set textContent(value) {
      this.children = value === "" || value == null ? [] : [new FakeText(value)];
    }
    get textContent() {
      return this.children.map((c) => c.textContent).join("");
    }
    appendChild(child) {
      this.children.push(child);
      return child;
    }
    append(...nodes) {
      for (const n of nodes) this.children.push(typeof n === "string" ? new FakeText(n) : n);
    }
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    }
    get firstChild() {
      return this.children[0] || null;
    }
    setAttribute(name, value) {
      this.attrs[name] = String(value);
    }
    getAttribute(name) {
      return name in this.attrs ? this.attrs[name] : null;
    }
    set href(value) {
      this.attrs.href = String(value);
    }
    get href() {
      return this.attrs.href;
    }
    addEventListener(type, fn) {
      (this.listeners[type] = this.listeners[type] || []).push(fn);
    }
    click() {
      for (const fn of this.listeners.click || []) fn({ preventDefault() {} });
    }
    insertBefore(node, before) {
      const i = before ? this.children.indexOf(before) : -1;
      if (i < 0) this.children.push(node);
      else this.children.splice(i, 0, node);
      return node;
    }
    focus() {}
    get options() {
      return this.children;
    }
    set value(v) {
      this._value = v;
    }
    get value() {
      if (this._value !== undefined) return this._value;
      return this.tag === "select" && this.children[0] ? this.children[0].value : "";
    }
  }
  for (const name of ["innerHTML", "outerHTML"]) {
    Object.defineProperty(FakeElement.prototype, name, {
      get() {
        throw new Error(`${name} was read`);
      },
      set() {
        throw new Error(`${name} was set`);
      },
    });
  }
  FakeElement.prototype.insertAdjacentHTML = function () {
    throw new Error("insertAdjacentHTML was called");
  };
  const byId = {};
  const doc = {
    created,
    byId,
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
    getElementById: (id) => (byId[id] = byId[id] || new FakeElement(id === "kb" ? "select" : "div")),
  };
  return doc;
}

function walk(node, visit) {
  visit(node);
  for (const child of node.children || []) walk(child, visit);
}
function textOf(node) {
  return node.textContent;
}
function elements(node) {
  const out = [];
  walk(node, (n) => n.nodeType === 1 && out.push(n));
  return out;
}

// ---- rendering ----------------------------------------------------------------------------

const IMG = '<img src=x onerror="window.__pwned=1">';
const SCRIPT = "<script>window.__pwned=1</script>";
const ALLOWED = new Set(["div", "span", "button", "p", "pre", "a", "strong", "ul", "li", "small"]);

function suiteRender() {
  const doc = makeDocument();
  const source = (n, extra = {}) => ({
    chunk_id: `kb:p${n}.md#0`,
    title_path: [`Page ${n}`, `Section ${IMG}`],
    source_rel: `p${n}.md`,
    source_url: `https://example.org/p${n}.md`,
    source_version: "v1",
    text: `Body ${n} ${IMG} and ${SCRIPT}`,
    untrusted: true,
    ...extra,
  });
  const response = {
    request_id: "req-1",
    status: "answered",
    found: true,
    answer: "",
    sentences: [
      { text: `Push it ${IMG}.`, citations: [{ marker: 1, chunk_id: "kb:p1.md#0" }] },
      { text: `Or ${SCRIPT} upload.`, citations: [{ marker: 2, chunk_id: "kb:p2.md#0" }] },
      { text: "No support.", citations: [] },
    ],
    sources: [source(1), source(2, { source_url: "javascript:alert(1)" })],
    citation_check: { removed_markers: [9, IMG], unsupported_sentences: 1, parse_attempts: 1 },
  };

  // The answer: sentences are text, markers are buttons, an unsupported sentence is marked.
  const answer = doc.createElement("div");
  const shown = [];
  const cites = R.renderAnswer(doc, answer, response, (marker, src) => shown.push([marker, src]));
  check(textOf(answer).includes(`Push it ${IMG}.`), "answer text with markup is shown as text");
  check(textOf(answer).includes(SCRIPT), "script text is shown as text");
  same(cites.map((c) => c.dataset.marker), ["1", "2"], "one button per citation");
  cites[1].click();
  same(shown.map((s) => s[0]), [2], "a click reports its marker");
  check(shown[0][1].chunk_id === "kb:p2.md#0", "a click reports its source");
  const unsupported = elements(answer).filter((e) => e.classes.has("unsupported"));
  same(unsupported.length, 1, "the sentence with no citation is marked");

  // The passage.
  const passage = doc.createElement("div");
  R.renderPassage(doc, passage, response.sources[0], 1);
  check(textOf(passage).includes(`Body 1 ${IMG} and ${SCRIPT}`), "passage text is text");
  check(textOf(passage).includes(`Section ${IMG}`), "title is text");
  const links = elements(passage).filter((e) => e.tag === "a");
  same(links.length, 1, "an https source url is a link");
  same(links[0].attrs.href, "https://example.org/p1.md", "the link goes where the url says");
  same(links[0].attrs.rel, "noopener noreferrer", "a link out of the page cannot reach it");

  const bad = doc.createElement("div");
  R.renderPassage(doc, bad, response.sources[1], 2);
  same(elements(bad).filter((e) => e.tag === "a").length, 0, "a javascript: url is no link");
  check(textOf(bad).includes("javascript:alert(1)"), "but it is shown, as text");

  // Removed markers, errors, notes, a streamed piece.
  const removed = doc.createElement("div");
  R.renderRemoved(doc, removed, response.citation_check.removed_markers);
  check(textOf(removed).includes("[9]"), "removed marker is listed");
  check(textOf(removed).includes(IMG), "even a marker that is not a number stays text");
  const err = doc.createElement("div");
  R.renderError(doc, err, { error: `x${IMG}`, request_id: `r${SCRIPT}`, message: "Internal error." });
  check(textOf(err).includes(`x${IMG}`) && textOf(err).includes(`r${SCRIPT}`), "error is text");
  const live = doc.createElement("div");
  R.appendText(doc, live, `Hel${IMG}`);
  R.appendText(doc, live, "lo");
  same(live.children.length, 2, "a streamed piece is one text node");
  check(live.children.every((c) => c.nodeType === 3), "and no element");
  R.clear(live);
  same(live.children.length, 0, "clear removes the children");

  const list = doc.createElement("div");
  const picked = [];
  R.renderSourceList(doc, list, response.sources, (src, n) => picked.push(n));
  check(textOf(list).includes(`Section ${IMG}`), "titles in the source list are text");
  elements(list).filter((e) => e.tag === "button")[1].click();
  same(picked, [2], "a source of the list can be picked");

  // Nothing but plain elements was ever created, and none carries an event handler attribute.
  for (const el of doc.created) {
    check(ALLOWED.has(el.tag), `element <${el.tag}> was created`);
    for (const name of Object.keys(el.attrs)) {
      check(!name.startsWith("on") && name !== "style" && name !== "srcdoc", `attribute ${name}`);
    }
  }
  check(!doc.created.some((e) => e.tag === "img" || e.tag === "script"), "no img, no script");
}

// ---- the page's own code, run against a fake document and a scripted server ----------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sse = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

// `script(url, init, calls)` answers a call with {status, headers, json} or
// {stream: [chunks], hang: bool} (a stream that never ends until the request is aborted).
function fakeServer(script) {
  const calls = [];
  async function fetchFake(url, init = {}) {
    calls.push({ url, init, body: init.body ? JSON.parse(init.body) : null });
    const reply = script(url, init, calls);
    if (reply.reject) throw new TypeError("Failed to fetch");
    if (reply.stream) {
      // The stream as the network delivers it: bytes, cut every `split` bytes (3 by default), so
      // a multi-byte character is often split between two reads.
      const bytes = new TextEncoder().encode(reply.stream.join(""));
      const size = reply.split || 3;
      const chunks = [];
      for (let i = 0; i < bytes.length; i += size) chunks.push(bytes.subarray(i, i + size));
      return {
        ok: true,
        status: 200,
        headers: { get: (name) => (name.toLowerCase() === "x-request-id" ? "req-fake" : null) },
        body: {
          getReader: () => ({
            read: async () => {
              if (chunks.length) return { done: false, value: chunks.shift() };
              if (!reply.hang) return { done: true };
              await new Promise((_, reject) => {
                init.signal.addEventListener("abort", () => {
                  const err = new Error("aborted");
                  err.name = "AbortError";
                  reject(err);
                });
              });
            },
          }),
        },
      };
    }
    return {
      ok: reply.status === undefined || reply.status < 400,
      status: reply.status || 200,
      headers: { get: () => (reply.headers || {})["x-request-id"] || null },
      json: async () => reply.json,
    };
  }
  fetchFake.calls = calls;
  return fetchFake;
}

async function startPage(script) {
  const doc = makeDocument();
  const storage = {};
  global.window = {
    RagRender: R,
    RagSse: { SseParser },
    sessionStorage: {
      getItem: (k) => (k in storage ? storage[k] : null),
      setItem: (k, v) => (storage[k] = String(v)),
    },
    crypto: { randomUUID: () => "11111111-2222-3333-4444-555555555555" },
  };
  global.document = doc;
  global.fetch = fakeServer((url, init, calls) => {
    if (url === "/v1/knowledge_bases") {
      return { json: { knowledge_bases: [{ kb: "demo", description: "Demo", available: true }] } };
    }
    return script(url, init, calls);
  });
  delete require.cache[require.resolve(path.join(web, "app.js"))];
  require(path.join(web, "app.js"));
  await sleep(10);
  return { doc, ui: (id) => doc.byId[id], fetch: global.fetch, storage };
}

async function ask(page, question) {
  page.ui("question").value = question;
  page.ui("ask").click();
  for (let i = 0; i < 100 && page.ui("ask").disabled; i++) await sleep(5);
}

const SOURCES = [1, 2, 3].map((n) => ({
  chunk_id: `kb:p${n}.md#0`,
  title_path: ["Guide", `Page ${n}`],
  source_rel: `p${n}.md`,
  source_url: `https://example.org/p${n}`,
  source_version: "v1",
  text: `Body of page ${n}`,
}));
const ANSWER = {
  request_id: "req-fake",
  status: "answered",
  found: true,
  answer: "x",
  sentences: [
    { text: "Push it.", citations: [{ marker: 1, chunk_id: SOURCES[0].chunk_id }] },
    { text: "Made up.", citations: [] },
  ],
  sources: SOURCES,
  nearest: [],
  notice: null,
  degraded: false,
  citation_check: { removed_markers: [7], unsupported_sentences: 1, parse_attempts: 1 },
};
const DONE = (id, status, request) => sse("done", { request_id: id, status, timings_ms: { request } });
const SESSION = "web-11111111-2222-3333-4444-555555555555";

async function suiteApp() {
  // 1. A streamed answer, then the passage behind a marker, then feedback.
  let page = await startPage(() => ({
    stream: [
      ": keep-alive\n\n",
      sse("progress", { node: "route", phase: "start" }),
      sse("progress", { node: "retrieve", phase: "end", round: 1, chunks: 3, chunks_total: 3, ms: 5 }),
      sse("progress", { node: "answer", phase: "start", round: 1 }),
      sse("token", { text: "Pus", attempt: 1 }),
      sse("token", { text: "h it [1].", attempt: 1 }),
      sse("mystery", { a: 1 }),
      "event: token\ndata: {not json\n\n",
      sse("answer", ANSWER),
      sse("done", {
        request_id: "req-fake",
        status: "answered",
        timings_ms: { request: 1500, first_progress: 12, first_token: 480 },
      }),
    ],
  }));
  same(page.ui("kb").children.map((o) => o.value), ["demo"], "the knowledge bases fill the menu");
  await ask(page, "How do I push?");
  const call = page.fetch.calls.find((c) => c.url === "/v1/ask_docs/stream");
  same(call.body, { kb: "demo", question: "How do I push?", session_id: SESSION }, "the request");
  same(call.init.method, "POST", "a POST");
  check(page.storage["rag.session"] === SESSION, "the session id is kept for the tab");
  const status = page.ui("status").textContent;
  check(status.startsWith("Done in 1.5 s. First progress after 12 ms, first text after 480 ms"), "status: " + status);
  check(page.ui("live").hidden, "the streamed text is replaced by the checked answer");
  check(page.ui("answer").textContent.includes("Push it. [1]"), "answer: " + page.ui("answer").textContent);
  check(page.ui("removed").textContent.includes("[7]") && !page.ui("removed").hidden, "removed marker");
  const cites = elements(page.ui("answer")).filter((e) => e.tag === "button");
  same(cites.length, 1, "one marker button");
  cites[0].click();
  const shown = page.ui("passage");
  check(shown.textContent.includes("Body of page 1") && !shown.hidden, "the passage opens");
  check(shown.textContent.startsWith("[1] Guide"), "with its marker and title: " + shown.textContent);
  elements(page.ui("sources")).filter((e) => e.tag === "button")[2].click();
  check(shown.textContent.includes("Body of page 3"), "a source of the list opens too");
  check(!page.ui("feedback").hidden, "the feedback buttons appear when the run is done");
  page.ui("not-useful").click();
  await sleep(10);
  const fb = page.fetch.calls.find((c) => c.url === "/v1/feedback");
  same(fb.body, { request_id: "req-fake", useful: false }, "the feedback request");
  check(page.ui("useful").disabled && page.ui("not-useful").disabled, "one verdict per answer");
  check(page.ui("feedback-status").textContent.startsWith("Thank you"), "feedback acknowledged");

  // 2. A question asked back; the reply goes with the resume token.
  let n = 0;
  page = await startPage(() => ({
    stream:
      ++n === 1
        ? [
            sse("progress", { node: "route", phase: "start" }),
            sse("need_clarification", {
              request_id: "r1",
              question: "Which library?",
              resume_token: "fake-resume-value-for-the-test",
              session_id: "s",
            }),
            DONE("r1", "need_clarification", 100),
          ]
        : [sse("answer", ANSWER), DONE("r2", "answered", 900)],
  }));
  await ask(page, "How do I fix this error?");
  check(!page.ui("clarify").hidden, "the clarification box is shown");
  same(page.ui("clarify-question").textContent, "Which library?", "the question asked back");
  check(page.ui("status").textContent.includes("one more detail"), "status says a detail is needed");
  check(page.ui("feedback").hidden, "no feedback buttons for a question asked back");
  page.ui("clarify-reply").value = "transformers";
  page.ui("clarify-send").click();
  for (let i = 0; i < 100 && page.ui("ask").disabled; i++) await sleep(5);
  const resumed = page.fetch.calls.filter((c) => c.url === "/v1/ask_docs/stream")[1].body;
  same(
    resumed,
    {
      kb: "demo",
      question: "How do I fix this error?",
      session_id: SESSION,
      resume_token: "fake-resume-value-for-the-test",
      answer: "transformers",
    },
    "the reply carries the token"
  );
  check(page.ui("clarify").hidden && page.ui("answer").textContent.includes("Push it."), "it goes on to the answer");

  // 3. An error event, a refused request, a cut stream, a retry, a cancel.
  page = await startPage(() => ({
    stream: [
      sse("progress", { node: "route", phase: "start" }),
      sse("error", { error: "retrieval_failed", request_id: "req-9", message: "The search backend failed." }),
    ],
  }));
  await ask(page, "q");
  const errorText = page.ui("error").textContent;
  check(errorText.includes("retrieval_failed") && errorText.includes("req-9"), "error event: " + errorText);
  check(!page.ui("ask").disabled, "the page can ask again after an error");

  page = await startPage(() => ({
    status: 404,
    headers: { "x-request-id": "req-8" },
    json: { error: "kb_not_found", request_id: "req-8", message: "No knowledge base." },
  }));
  await ask(page, "q");
  const refused = page.ui("error").textContent;
  check(refused.includes("kb_not_found") && refused.includes("req-8"), "refused request: " + refused);

  page = await startPage(() => ({ stream: [sse("progress", { node: "route", phase: "start" })] }));
  await ask(page, "q");
  check(page.ui("status").textContent.includes("ended before the answer was complete"), "a cut stream");

  page = await startPage(() => ({
    stream: [
      sse("token", { text: "first try", attempt: 1 }),
      sse("progress", { node: "answer", phase: "retry", attempt: 2 }),
      sse("token", { text: "second try", attempt: 2 }),
    ],
  }));
  await ask(page, "q");
  same(page.ui("live").textContent, "second try", "a retry clears the text of the first attempt");

  page = await startPage(() => ({ stream: [sse("progress", { node: "route", phase: "start" })], hang: true }));
  page.ui("question").value = "slow";
  page.ui("ask").click();
  await sleep(20);
  check(!page.ui("cancel").hidden && page.ui("ask").disabled, "while running: cancel shown, ask disabled");
  const signal = page.fetch.calls.find((c) => c.url === "/v1/ask_docs/stream").init.signal;
  check(signal.aborted === false, "the request is open");
  page.ui("cancel").click();
  check(signal.aborted === true, "cancel closes the connection (AbortController)");
  await sleep(20);
  check(page.ui("status").textContent.startsWith("Cancelled"), "after cancel: " + page.ui("status").textContent);
  check(!page.ui("ask").disabled && page.ui("cancel").hidden, "the page is ready again");

  // 3b. Text that is not ASCII, cut between bytes: the page decodes the stream as a stream.
  const CJK = "\u4f60\u597d\u4e16\u754c \ud83d\ude00";
  page = await startPage(() => ({
    stream: [
      sse("token", { text: CJK, attempt: 1 }),
      sse("answer", {
        ...ANSWER,
        sentences: [{ text: CJK, citations: [{ marker: 1, chunk_id: SOURCES[0].chunk_id }] }],
      }),
      DONE("r", "answered", 1),
    ],
    split: 1,
  }));
  await ask(page, "q");
  check(page.ui("answer").textContent.includes(CJK), "non-ASCII answer intact: " + page.ui("answer").textContent);
  check(!page.ui("answer").textContent.includes("\ufffd"), "no replacement character in the answer");
  page = await startPage(() => ({ stream: [sse("token", { text: CJK, attempt: 1 })], split: 2 }));
  await ask(page, "q");
  same(page.ui("live").textContent, CJK, "non-ASCII streamed text intact");

  // 3c. A run that fails after text was streamed: the text stays, marked as not checked.
  page = await startPage(() => ({
    stream: [
      sse("token", { text: "Half an answer", attempt: 1 }),
      sse("error", { error: "internal_error", request_id: "req-7", message: "Internal error." }),
    ],
  }));
  await ask(page, "q");
  check(page.ui("live").textContent === "Half an answer" && !page.ui("live").hidden, "the streamed text stays");
  check(page.ui("live").classes.has("unverified"), "and is marked unverified");
  check(page.ui("notice").textContent.includes("not checked") && !page.ui("notice").hidden, "with a note");
  page = await startPage(() => ({ stream: [sse("token", { text: "Cut off", attempt: 1 })] }));
  await ask(page, "q");
  check(page.ui("live").classes.has("unverified") && page.ui("live").textContent === "Cut off", "a cut stream too");
  page = await startPage(() => ({ stream: [sse("token", { text: "Fine", attempt: 1 }), sse("answer", ANSWER), DONE("r", "answered", 1)] }));
  await ask(page, "q");
  check(!page.ui("live").classes.has("unverified") && page.ui("notice").hidden, "a good run is not marked");

  // 3d. A reply to a question asked back that fails or is cancelled can be sent again.
  const ASK_BACK = [
    sse("need_clarification", { request_id: "r1", question: "Which library?", resume_token: "fake-resume-value-for-the-test", session_id: "s" }),
    DONE("r1", "need_clarification", 100),
  ];
  const resumeCalls = (p) => p.fetch.calls.filter((c) => c.url === "/v1/ask_docs/stream").slice(1);
  for (const [label, failure] of [
    ["a connection that fails", { reject: true }],
    ["an error event", { stream: [sse("error", { error: "internal_error", request_id: "r2", message: "Internal error." })] }],
    ["an HTTP error", { status: 502, headers: { "x-request-id": "r3" }, json: { error: "retrieval_failed", request_id: "r3", message: "The search backend failed." } }],
  ]) {
    let step = 0;
    page = await startPage(() => (++step === 1 ? { stream: ASK_BACK } : step === 2 ? failure : { stream: [sse("answer", ANSWER), DONE("r4", "answered", 5)] }));
    await ask(page, "How do I fix this error?");
    page.ui("clarify-reply").value = "transformers";
    page.ui("clarify-send").click();
    for (let i = 0; i < 100 && page.ui("ask").disabled; i++) await sleep(5);
    check(!page.ui("clarify").hidden, label + ": the question stays on the page");
    same(page.ui("clarify-reply").value, "transformers", label + ": the reply stays");
    check(!page.ui("clarify-send").disabled, label + ": it can be sent again");
    check(!page.ui("error").hidden, label + ": the error is shown");
    page.ui("clarify-send").click();
    for (let i = 0; i < 100 && page.ui("ask").disabled; i++) await sleep(5);
    const again = resumeCalls(page);
    same(again.length, 2, label + ": a second reply was sent");
    same(again[1] && again[1].body, again[0] && again[0].body, label + ": with the same token and the same reply");
    check(page.ui("clarify").hidden && page.ui("answer").textContent.includes("Push it."), label + ": then it ends with the answer");
  }
  // Cancelled while the reply is running: the question stays too.
  let calls3 = 0;
  page = await startPage(() => (++calls3 === 1 ? { stream: ASK_BACK } : { stream: [sse("progress", { node: "route", phase: "start" })], hang: true }));
  await ask(page, "How do I fix this error?");
  page.ui("clarify-reply").value = "transformers";
  page.ui("clarify-send").click();
  await sleep(20);
  check(!page.ui("clarify").hidden, "while the reply runs the question stays visible");
  page.ui("cancel").click();
  await sleep(20);
  check(!page.ui("clarify").hidden && page.ui("clarify-reply").value === "transformers", "cancelled: question and reply stay");
  // A token the server no longer accepts is dropped, with the error.
  let calls4 = 0;
  page = await startPage(() => (++calls4 === 1 ? { stream: ASK_BACK } : { status: 400, headers: { "x-request-id": "r5" }, json: { error: "invalid_resume_token", request_id: "r5", message: "Not valid." } }));
  await ask(page, "How do I fix this error?");
  page.ui("clarify-reply").value = "transformers";
  page.ui("clarify-send").click();
  for (let i = 0; i < 100 && page.ui("ask").disabled; i++) await sleep(5);
  check(page.ui("clarify").hidden && page.ui("error").textContent.includes("invalid_resume_token"), "a used up token is dropped");

  // 4. A new conversation gets a new session id.
  page = await startPage(() => ({ stream: [DONE("r", "answered", 1)] }));
  await ask(page, "one");
  const before = page.storage["rag.session"];
  global.window.crypto.randomUUID = () => "99999999-8888-7777-6666-555555555555";
  page.ui("new-chat").click();
  const after = page.storage["rag.session"];
  check(after !== before && /^web-[A-Za-z0-9._-]+$/.test(after), "a new conversation, a new session id");
}

const suite = process.argv[2];
async function main() {
  if (suite === "sse") suiteSse();
  else if (suite === "render") suiteRender();
  else if (suite === "app") await suiteApp();
  else {
    console.error("usage: ui_test.js sse|render|app");
    process.exit(2);
  }
  if (failures.length) {
    console.log(failures.map((f) => "FAIL " + f).join("\n"));
    process.exit(1);
  }
  console.log(`${suite}: ok`);
}
main().catch((error) => {
  console.log("FAIL exception " + (error && error.stack ? error.stack : error));
  process.exit(1);
});
