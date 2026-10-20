// Parser of a Server-Sent Events stream (spec P3-04). The browser's EventSource cannot send a
// POST, so the page reads the response of `fetch` itself and feeds the decoded text to this
// parser. It follows the line rules of the SSE specification: lines end with LF, CR or CRLF;
// a line that starts with a colon is a comment (the server's heartbeat) and is skipped; `event:`
// names the event, each `data:` line adds one line of data (joined with a newline); a blank
// line dispatches the event if it has data. `id:` and `retry:` are ignored.
(function (root) {
  "use strict";

  const DEFAULT_MAX_BYTES = 1024 * 1024;

  class SseParser {
    constructor(options) {
      this.maxBytes = (options && options.maxBytes) || DEFAULT_MAX_BYTES;
      this.buffer = ""; // the part of a line that has not ended yet
      this.skipLf = false; // the last chunk ended with a CR: a LF at the start belongs to it
      this.eventName = "";
      this.dataLines = [];
      this.dataSize = 0;
    }

    // Feed decoded text; returns the events it completed, as {event, data} (data is a string).
    push(chunk) {
      if (this.skipLf && chunk.length > 0) {
        if (chunk[0] === "\n") chunk = chunk.slice(1);
        this.skipLf = false;
      }
      this.buffer += chunk;
      const events = [];
      const lineEnd = /\r\n?|\n/g;
      let start = 0;
      let match;
      while ((match = lineEnd.exec(this.buffer)) !== null) {
        this.line(this.buffer.slice(start, match.index), events);
        start = match.index + match[0].length;
        if (match[0] === "\r" && start === this.buffer.length) this.skipLf = true;
      }
      this.buffer = this.buffer.slice(start);
      if (this.buffer.length > this.maxBytes) {
        throw new Error("a line of the event stream is longer than the limit");
      }
      return events;
    }

    line(text, events) {
      if (text === "") {
        if (this.dataLines.length > 0) {
          events.push({ event: this.eventName || "message", data: this.dataLines.join("\n") });
        }
        this.eventName = "";
        this.dataLines = [];
        this.dataSize = 0;
        return;
      }
      if (text[0] === ":") return; // a comment: the heartbeat
      const colon = text.indexOf(":");
      const field = colon < 0 ? text : text.slice(0, colon);
      let value = colon < 0 ? "" : text.slice(colon + 1);
      if (value[0] === " ") value = value.slice(1);
      if (field === "event") {
        this.eventName = value;
      } else if (field === "data") {
        this.dataSize += value.length + 1;
        if (this.dataSize > this.maxBytes) throw new Error("an event is longer than the limit");
        this.dataLines.push(value);
      }
    }
  }

  const api = { SseParser };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RagSse = api;
})(typeof window !== "undefined" ? window : globalThis);
