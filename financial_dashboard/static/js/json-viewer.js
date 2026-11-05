// Enhances audit JSON blocks with a collapsible tree, pretty, minify, and copy controls.
(function () {
    "use strict";

    const SHORT_STRING_LIMIT = 200;
    const PREVIEW_LIMIT = 90;
    const COPY_LABEL_MS = 1200;

    /** Build a span element with one class and text. */
    function span(className, text) {
        const node = document.createElement("span");
        node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    /** Build a div element with one class. */
    function div(className) {
        const node = document.createElement("div");
        node.className = className;
        return node;
    }

    /** Build a toolbar button. */
    function button(label) {
        const node = document.createElement("button");
        node.type = "button";
        node.className = "jv-btn";
        node.textContent = label;
        return node;
    }

    /** Add the key label to a row or summary when a key exists. */
    function appendKey(parent, key, isArrayIndex) {
        if (key === null) return;
        parent.append(span("jv-key", isArrayIndex ? "[" + key + "]: " : JSON.stringify(key) + ": "));
    }

    /** Return the text preview for a long or multiline string. */
    function previewLine(value) {
        const firstLine = value.split("\n")[0];
        return firstLine.length > PREVIEW_LIMIT ? firstLine.slice(0, PREVIEW_LIMIT) + "…" : firstLine;
    }

    /** Build a folded text node for a long or multiline string. */
    function renderTextNode(key, value, isArrayIndex) {
        const details = document.createElement("details");
        details.className = "jv-text";
        const lineCount = value.split("\n").length;
        details.open = lineCount <= 3 && value.length <= 320;
        const summary = document.createElement("summary");
        summary.className = "jv-summary";
        appendKey(summary, key, isArrayIndex);
        summary.append(span("jv-string jv-preview", previewLine(value)));
        const meta = span("jv-meta", lineCount > 1 ? "(" + lineCount + " lines)" : "(" + value.length + " chars)");
        summary.append(meta);
        details.append(summary);
        const body = div("jv-multiline");
        body.textContent = value;
        details.append(body);
        return details;
    }

    /** Build a row for one scalar value. */
    function renderScalar(key, value, isArrayIndex) {
        const row = div("jv-row");
        appendKey(row, key, isArrayIndex);
        if (typeof value === "string") row.append(span("jv-string", JSON.stringify(value)));
        else if (typeof value === "number") row.append(span("jv-number", String(value)));
        else if (typeof value === "boolean") row.append(span("jv-bool", String(value)));
        else row.append(span("jv-null", String(value)));
        return row;
    }

    /** Build a collapsible node for a non-empty object or array. */
    function renderContainer(key, value, isArrayIndex, depth) {
        const isArray = Array.isArray(value);
        const entries = isArray ? value.map((item, index) => [index, item]) : Object.entries(value);
        const details = document.createElement("details");
        details.className = "jv-node";
        details.open = depth < 2;
        const summary = document.createElement("summary");
        summary.className = "jv-summary";
        appendKey(summary, key, isArrayIndex);
        const count = entries.length;
        const label = isArray
            ? "[" + count + (count === 1 ? " item]" : " items]")
            : "{" + count + (count === 1 ? " key}" : " keys}");
        summary.append(span("jv-meta", label));
        details.append(summary);
        const children = div("jv-children");
        entries.forEach(([childKey, childValue]) => {
            children.append(renderNode(childKey, childValue, depth + 1, isArray));
        });
        details.append(children);
        return details;
    }

    /** Render one JSON value as a tree node. */
    function renderNode(key, value, depth, isArrayIndex) {
        if (value !== null && typeof value === "object") {
            const size = Array.isArray(value) ? value.length : Object.keys(value).length;
            if (size === 0) {
                const row = div("jv-row");
                appendKey(row, key, isArrayIndex);
                row.append(span("jv-bracket", Array.isArray(value) ? "[]" : "{}"));
                return row;
            }
            return renderContainer(key, value, isArrayIndex, depth);
        }
        if (typeof value === "string" && (value.includes("\n") || value.length > SHORT_STRING_LIMIT)) {
            return renderTextNode(key, value, isArrayIndex);
        }
        return renderScalar(key, value, isArrayIndex);
    }

    /** Return the document details inside one element. */
    function detailsIn(root) {
        return Array.from(root.querySelectorAll("details"));
    }

    /** Wire the toolbar and tree for one JSON block. */
    function enhance(pre) {
        const originalPretty = pre.textContent;
        let parsed;
        try {
            parsed = JSON.parse(pre.dataset.tree);
        } catch (error) {
            return;
        }
        const treeEl = document.createElement("div");
        treeEl.className = "jv-tree";
        treeEl.append(renderNode(null, parsed, 0, false));

        let mode = "tree";
        const treeButton = button("Tree");
        const prettyButton = button("Pretty");
        const minifyButton = button("Minify");
        const expandButton = button("Expand all");
        const collapseButton = button("Collapse all");
        const copyButton = button("Copy");
        const modeButtons = [["tree", treeButton], ["pretty", prettyButton], ["minify", minifyButton]];

        /** Select one display mode and update the visible panes. */
        function setMode(next) {
            mode = next;
            treeEl.hidden = next !== "tree";
            pre.hidden = next === "tree";
            modeButtons.forEach(([name, node]) => node.classList.toggle("is-active", name === next));
        }

        treeButton.addEventListener("click", () => setMode("tree"));
        prettyButton.addEventListener("click", () => {
            pre.textContent = originalPretty;
            setMode("pretty");
        });
        minifyButton.addEventListener("click", () => {
            pre.textContent = pre.dataset.compact;
            setMode("minify");
        });
        expandButton.addEventListener("click", () => {
            setMode("tree");
            detailsIn(treeEl).forEach((node) => {
                node.open = true;
            });
        });
        collapseButton.addEventListener("click", () => {
            setMode("tree");
            detailsIn(treeEl).forEach((node) => {
                node.open = false;
            });
            const root = treeEl.querySelector("details");
            if (root) root.open = true;
        });
        copyButton.addEventListener("click", () => {
            const text = mode === "minify" ? pre.textContent : originalPretty;
            const flash = (label) => {
                copyButton.textContent = label;
                window.setTimeout(() => {
                    copyButton.textContent = "Copy";
                }, COPY_LABEL_MS);
            };
            navigator.clipboard.writeText(text).then(
                () => flash("Copied"),
                () => flash("Copy failed"),
            );
        });

        const toolbar = div("jv-toolbar");
        toolbar.append(treeButton, prettyButton, minifyButton, expandButton, collapseButton);
        // The clipboard API exists only on https or localhost.
        if (navigator.clipboard) toolbar.append(copyButton);

        const wrapper = div("jv");
        pre.replaceWith(wrapper);
        pre.hidden = true;
        wrapper.append(toolbar, treeEl, pre);
        treeButton.classList.add("is-active");
    }

    document.querySelectorAll("pre[data-json-viewer]").forEach(enhance);
})();
