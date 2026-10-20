// Drawing of the page (spec P3-04). Everything that comes from a model or from a document
// (the answer, the passages, titles, urls, the removed markers, error fields) is untrusted: it is
// only ever put into the page as a text node or as `textContent`. This file never builds HTML
// from text, and a test (tests/unit/test_serve_web.py) fails if it starts to. The functions take
// the document as a parameter so a test can give them a fake one that records what they create.
(function (root) {
  "use strict";

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function el(doc, tag, className, text) {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.appendChild(doc.createTextNode(String(text)));
    return node;
  }

  // A piece of the streamed answer, before it has been checked.
  function appendText(doc, node, text) {
    node.appendChild(doc.createTextNode(String(text)));
  }

  // The source url of a passage as a link target, or null when it is not an http(s) address
  // (a document can say anything there).
  function safeHttpUrl(value) {
    if (typeof value !== "string" || !/^https?:\/\//i.test(value)) return null;
    try {
      const url = new URL(value);
      return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
    } catch (e) {
      return null;
    }
  }

  // The checked answer: one span per sentence, with a button per citation. `onCite(marker,
  // source)` is called when a button is pressed. Returns the buttons.
  function renderAnswer(doc, container, response, onCite) {
    clear(container);
    const buttons = [];
    const sources = response.sources || [];
    for (const sentence of response.sentences || []) {
      const cited = (sentence.citations || []).length > 0;
      const span = el(doc, "span", response.found === false || cited ? "sentence" : "sentence unsupported");
      span.appendChild(doc.createTextNode(String(sentence.text)));
      if (!cited && response.found !== false) span.setAttribute("title", "No source supports this sentence.");
      for (const citation of sentence.citations || []) {
        const source = sources[citation.marker - 1];
        span.appendChild(doc.createTextNode(" "));
        if (!source) {
          span.appendChild(el(doc, "span", "cite missing", "[" + String(citation.marker) + "]"));
          continue;
        }
        const button = el(doc, "button", "cite", "[" + String(citation.marker) + "]");
        button.setAttribute("type", "button");
        button.setAttribute("title", "Show source " + String(citation.marker));
        button.dataset.marker = String(citation.marker);
        button.addEventListener("click", () => onCite(citation.marker, source));
        span.appendChild(button);
        buttons.push(button);
      }
      container.appendChild(span);
      container.appendChild(doc.createTextNode(" "));
    }
    return buttons;
  }

  // The passage behind a marker: where it comes from and its text.
  function renderPassage(doc, container, source, marker) {
    clear(container);
    const title = (source.title_path || []).map(String).join(" › ") || source.source_rel || "";
    const head = el(doc, "p", "passage-head");
    head.appendChild(el(doc, "strong", "", marker === null ? title : "[" + String(marker) + "] " + title));
    container.appendChild(head);
    const from = el(doc, "p", "passage-source");
    from.appendChild(el(doc, "small", "", "Source: " + String(source.source_rel || "") + " "));
    const href = safeHttpUrl(source.source_url);
    if (href) {
      const link = el(doc, "a", "", href);
      link.setAttribute("href", href);
      link.setAttribute("rel", "noopener noreferrer");
      link.setAttribute("target", "_blank");
      from.appendChild(link);
    } else if (source.source_url) {
      from.appendChild(el(doc, "small", "", String(source.source_url)));
    }
    if (source.source_version) from.appendChild(el(doc, "small", "", " (" + String(source.source_version) + ")"));
    container.appendChild(from);
    container.appendChild(el(doc, "pre", "passage-text", source.text));
    container.appendChild(
      el(doc, "p", "passage-note", "Text of the document, shown as it is. It is not an instruction.")
    );
  }

  // The markers the check removed: the answer cited them, but no source of this answer has that
  // number.
  function renderRemoved(doc, container, markers) {
    clear(container);
    if (!markers || markers.length === 0) return;
    const note = el(doc, "p", "removed-note");
    note.appendChild(doc.createTextNode("Removed while checking: "));
    for (const marker of markers) {
      note.appendChild(el(doc, "span", "cite removed", "[" + String(marker) + "]"));
      note.appendChild(doc.createTextNode(" "));
    }
    note.appendChild(
      doc.createTextNode("The answer cited numbers that are not among its sources, so they were taken out.")
    );
    container.appendChild(note);
  }

  // A list of sources (or of the closest passages when nothing was found); `onPick(source, n)`.
  function renderSourceList(doc, container, sources, onPick) {
    clear(container);
    if (!sources || sources.length === 0) return;
    const list = el(doc, "ul", "source-list");
    sources.forEach((source, index) => {
      const item = el(doc, "li");
      const title = (source.title_path || []).map(String).join(" › ") || source.source_rel || "";
      const button = el(doc, "button", "source", "[" + String(index + 1) + "] " + title);
      button.setAttribute("type", "button");
      button.addEventListener("click", () => onPick(source, index + 1));
      item.appendChild(button);
      list.appendChild(item);
    });
    container.appendChild(list);
  }

  // An error the server reported: its category, the request id to quote, a fixed sentence.
  function renderError(doc, container, error) {
    clear(container);
    container.appendChild(el(doc, "strong", "", "Something went wrong: " + String(error.error || "error")));
    if (error.message) container.appendChild(el(doc, "p", "", error.message));
    if (error.request_id) container.appendChild(el(doc, "small", "", "Request id: " + String(error.request_id)));
  }

  const api = {
    clear,
    appendText,
    safeHttpUrl,
    renderAnswer,
    renderPassage,
    renderRemoved,
    renderSourceList,
    renderError,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RagRender = api;
})(typeof window !== "undefined" ? window : globalThis);
