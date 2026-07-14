// The cashflow page's two charts: a trailing-months trend and a breakdown of
// whichever tile is selected. Both are hand-rolled inline SVG over the shared
// chart machinery; the page's own markup is what configures them.
//
// Where the numbers come from differs on purpose, and it is the whole reason the
// breakdown carries no fetch:
//
//   * The breakdown draws the *selected range*, which the server already
//     aggregated to render the tables — so it is handed that summary, serialized
//     into the page. Fetching /api/cashflow/summary here would re-run the same
//     aggregate queries for numbers already in the document.
//   * The trend is a trailing-window series that ignores the selected range, so
//     it genuinely is not on the page, and it is fetched.

import {
    bindHover,
    esc,
    fmtINR,
    fmtMonth,
    jsonBlock,
    nearestByX,
    onResize,
    placeTooltip,
    sizedRenderer,
    toViewBox,
    tooltipMarkup,
} from "./charts.js";

// ---------------------------------------------------------------------------
// Trend: grouped income / expense / net-invested bars per month.
// ---------------------------------------------------------------------------
const trend = async () => {
    const svg = document.getElementById("cf-trend");
    const caption = document.getElementById("cf-trend-caption");
    const empty = document.getElementById("cf-trend-empty");
    if (!svg) return;

    // Server-computed bounds, one entry per month drawn. Clicking a month sets
    // the report's range to it, which is a plain navigation to /cashflow with
    // those bounds — so each month is an <a>, and a month with no bounds simply
    // gets no link rather than a broken one.
    const RANGES = jsonBlock("cf-trend-ranges");

    let points;
    try {
        const response = await fetch(svg.dataset.trendUrl, { cache: "no-store" });
        if (!response.ok) return;
        points = await response.json();
    } catch (error) {
        console.debug("Failed to load cashflow trend", error);
        return;
    }

    const series = [
        { key: "income", label: "Income", color: "var(--clr-credit)" },
        { key: "expense", label: "Expense", color: "var(--clr-debit)" },
        { key: "net_invested", label: "Net invested", color: "currentColor" },
    ];
    // salary_count is kept, not dropped: a calendar month holds 2 or 3 paychecks
    // on a ~14-day cycle, so a month's income swings by half with nothing at all
    // having changed. The tooltip's paycheck count is what lets a reader tell
    // that apart from a real swing.
    const values = (points || []).map((p) => ({
        month: p.month,
        income: Number(p.income),
        expense: Number(p.expense),
        net_invested: Number(p.net_invested),
        salary_count: Number(p.salary_count),
    }));

    // The trend endpoint pre-seeds every month in the window, so an empty history
    // comes back as a full array of zeros, never as an empty one: gating the
    // empty state on the array's length alone would make it unreachable and draw
    // a row of zero-height bars instead. Both are checked, but the all-zero test
    // is the one that fires.
    const allZero = values.every((v) => !v.income && !v.expense && !v.net_invested);
    if (!values.length || allZero) {
        empty.style.display = "block";
        svg.style.display = "none";
        caption.style.display = "none";
        return;
    }

    // Bars are drawn from a zero baseline, so the scale must span both signs: a
    // redemption-heavy month has a negative net invested.
    const flat = values.flatMap((v) => [v.income, v.expense, v.net_invested]);
    const max = Math.max(0, ...flat);
    const min = Math.min(0, ...flat);
    const span = max - min || 1;

    const height = 170;
    const padTop = 10;
    const padBottom = 26;
    const padLeft = 44;
    const padRight = 8;
    let bars = [];   // [x, y, w, h, monthIndex, seriesIndex]
    let width = 600;

    const draw = (rawWidth) => {
        width = rawWidth;
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        const plotW = width - padLeft - padRight;
        const plotH = height - padTop - padBottom;
        const yOf = (v) => padTop + plotH * (1 - (v - min) / span);
        const zeroY = yOf(0);

        const slot = plotW / values.length;
        const groupW = Math.max(slot * 0.72, 3);
        const barW = Math.max(groupW / series.length, 1);

        bars = [];
        let barsSvg = "";
        values.forEach((v, i) => {
            const groupX = padLeft + slot * i + (slot - groupW) / 2;
            series.forEach((s, j) => {
                const value = v[s.key];
                const y = Math.min(yOf(value), zeroY);
                const h = Math.max(Math.abs(yOf(value) - zeroY), value === 0 ? 0 : 0.75);
                const x = groupX + barW * j;
                bars.push([x, y, barW, h, i, j]);
                barsSvg +=
                    `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(barW - 1, 0.75).toFixed(1)}" height="${h.toFixed(1)}" fill="${s.color}" opacity="0.85" />`;
            });
        });

        // Deduped: an all-zero window collapses max/mid/min onto one value, and
        // three stacked "₹0" labels on one gridline read as a rendering fault
        // rather than as an empty month.
        const ticks = [...new Set([max, (max + min) / 2, min])].map((t) =>
            `<text x="${padLeft - 5}" y="${(yOf(t) + 3).toFixed(1)}" text-anchor="end" font-size="9" fill="currentColor" opacity="0.55">${fmtINR(t)}</text>` +
            `<line x1="${padLeft}" x2="${width - padRight}" y1="${yOf(t).toFixed(1)}" y2="${yOf(t).toFixed(1)}" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="2 3" />`
        ).join("");
        const zeroLine = `<line x1="${padLeft}" x2="${width - padRight}" y1="${zeroY.toFixed(1)}" y2="${zeroY.toFixed(1)}" stroke="currentColor" stroke-opacity="0.25" />`;
        const xLabels =
            `<text x="${padLeft}" y="${height - 12}" text-anchor="start" font-size="9" fill="currentColor" opacity="0.55">${fmtMonth(values[0].month)}</text>` +
            `<text x="${width - padRight}" y="${height - 12}" text-anchor="end" font-size="9" fill="currentColor" opacity="0.55">${fmtMonth(values[values.length - 1].month)}</text>`;
        const legend = series.map((s, j) =>
            `<rect x="${(padLeft + j * 96).toFixed(1)}" y="${height - 8}" width="7" height="7" fill="${s.color}" opacity="0.85" />` +
            `<text x="${(padLeft + j * 96 + 11).toFixed(1)}" y="${height - 2}" font-size="9" fill="currentColor" opacity="0.55">${s.label}</text>`
        ).join("");

        // One transparent, full-height hit area per month, laid over the bars and
        // linking to that month's range: clicking a month selects it. Pointer
        // events still reach the SVG's hover handler by bubbling, so the tooltip
        // keeps working through them.
        const monthLinks = values.map((v, i) => {
            const range = RANGES[v.month];
            if (!range) return "";
            const x = padLeft + slot * i;
            return (
                `<a href="/cashflow?date_from=${esc(range[0])}&amp;date_to=${esc(range[1])}" data-month="${esc(v.month)}" aria-label="Set the range to ${esc(fmtMonth(v.month))}">` +
                `<rect x="${x.toFixed(1)}" y="${padTop}" width="${slot.toFixed(1)}" height="${plotH.toFixed(1)}" fill="transparent" style="cursor: pointer;" />` +
                `</a>`
            );
        }).join("");

        const hover =
            `<g id="cf-trend-hover" style="opacity: 0; pointer-events: none; transition: opacity 0.1s;">` +
            tooltipMarkup("cf-trend-hover") +
            `</g>`;

        svg.innerHTML = ticks + zeroLine + barsSvg + monthLinks + xLabels + legend + hover;
    };

    const render = sizedRenderer(svg, draw);
    render();

    // The series come from the same bank-scoped, cash-basis aggregation the tiles
    // do, so the caption says so: a chart that quietly counted card swipes while
    // the tiles counted card bills would contradict them month by month, and a
    // reader would have no way to see which of the two they were looking at.
    const totals = series.map((s) => values.reduce((acc, v) => acc + v[s.key], 0));
    caption.innerHTML =
        `<span>${fmtMonth(values[0].month)} &rarr; ${fmtMonth(values[values.length - 1].month)} &middot; bank accounts, cash basis</span>` +
        `<span>` + series.map((s, j) => `${s.label}: <strong>${fmtINR(totals[j])}</strong>`).join(" &middot; ") +
        ` &middot; <span class="text-sm text-muted">hover to inspect &middot; click a month to set the range</span></span>`;

    const showHover = (clientX, clientY) => {
        if (!bars.length) return;
        const point = toViewBox(svg, clientX, clientY);
        if (!point) return;

        // Nearest by x only: the bars in a group are adjacent and thin, so
        // horizontal distance alone picks the one under the cursor.
        const nearest = nearestByX(bars, point.x, (bar) => bar[0] + bar[2] / 2);
        if (nearest < 0) return;
        const [bx, by, bw, , mi, si] = bars[nearest];
        const v = values[mi];
        // The paycheck count rides in the tooltip because it is what explains the
        // month: a 3-paycheck month's income is not a raise.
        const paychecks = `${v.salary_count} paycheck${v.salary_count === 1 ? "" : "s"}`;
        const text = `${fmtMonth(v.month)} · ${series[si].label} · ${fmtINR(v[series[si].key])} · ${paychecks}`;

        // Re-queried, not held: every draw rebuilds the SVG's innerHTML.
        const group = svg.querySelector("#cf-trend-hover");
        if (!group) return;
        placeTooltip({
            label: svg.querySelector("#cf-trend-hover-label"),
            bg: svg.querySelector("#cf-trend-hover-bg"),
            text,
            anchorX: bx + bw / 2,
            anchorY: by,
            leftEdge: padLeft,
            rightEdge: width - padRight,
            drawableW: Math.max(width - padLeft - padRight, 20),
            padTop,
            aboveGap: 6,
            belowGap: 6,
        });
        group.style.opacity = "1";
    };
    const hideHover = () => {
        const g = svg.querySelector("#cf-trend-hover");
        if (g) g.style.opacity = "0";
    };

    bindHover(svg, showHover, hideHover);
    onResize(svg, render);
};

// ---------------------------------------------------------------------------
// Breakdown: the *selected* tile's lines, one diverging bar each, drawn from a
// zero axis so a contra credit (refund, cashback, redemption) reads as the
// leftward bar it is. Each bar is an <a> into the matching drill-through. All
// five tiles are selectable, which is the only way the Transfers-in and
// Uncategorized lines can be charted at all.
// ---------------------------------------------------------------------------
const breakdown = () => {
    const svg = document.getElementById("cf-breakdown");
    const caption = document.getElementById("cf-breakdown-caption");
    const empty = document.getElementById("cf-breakdown-empty");
    const title = document.getElementById("cf-breakdown-title");
    const tiles = [...document.querySelectorAll("[data-select]")];
    if (!svg) return;

    // The summary the server rendered this page from, serialized into it. Not
    // fetched: it is the same range, so a fetch would be the same aggregate
    // queries run twice for one page load. Amounts are Decimal strings, exactly
    // as the JSON API sends them, so each one is coerced before arithmetic.
    const summary = jsonBlock("cf-summary", null);
    if (!summary) return;

    // Each tile: the label it charts under, the lines it charts, the key that
    // identifies one of those lines, and the bar colour.
    //
    // An investment slug is direction-split — the same slug is one line as a
    // contribution and another as a redemption — so its key carries the kind;
    // transfers-in lines are one slug grouped by counterparty, so theirs is the
    // counterparty. A key of "" is the line for rows that carry no counterparty /
    // no category at all.
    const TILES = {
        income: {
            label: "Income",
            lines: (s) => s.income.lines,
            key: (l) => l.slug ?? "",
            color: "var(--clr-credit)",
        },
        expense: {
            label: "Expense",
            lines: (s) => s.expense.lines,
            key: (l) => l.slug ?? "",
            color: "var(--clr-debit)",
        },
        net_invested: {
            label: "Net Invested",
            lines: (s) => s.investment.lines,
            key: (l) => `${l.slug}:${l.kind}`,
            color: "currentColor",
        },
        transfers_in: {
            label: "Transfers In",
            lines: (s) => s.transfers_in.lines,
            key: (l) => l.counterparty ?? "",
            color: "var(--clr-credit)",
        },
        uncategorized: {
            label: "Uncategorized",
            lines: (s) => s.uncategorized.lines,
            key: (l) => l.slug ?? "",
            color: "currentColor",
        },
    };

    // A bar's link is the one the server already rendered for that line in the
    // table below, looked up by tile and key — not rebuilt here. The rules differ
    // per tile (a currency clause on the rupee buckets, a direction on a split
    // investment slug, a counterparty on a transfer, a category_null on the line
    // for rows with no category), and a second copy of them in JavaScript is a
    // second thing to drift.
    //
    // The maps are null-prototyped: a key is a category slug or a raw
    // counterparty, i.e. arbitrary text out of the ledger, and on a plain object a
    // row keyed "__proto__" would read back as Object.prototype — a bar linking to
    // "[object Object]" instead of to its rows.
    const DRILL = Object.create(null);
    document.querySelectorAll("a[data-line]").forEach((a) => {
        (DRILL[a.dataset.line] ??= Object.create(null))[a.dataset.key] =
            a.getAttribute("href");
    });

    const rowsFor = (name) => {
        const tile = TILES[name];
        return tile.lines(summary)
            .map((l) => ({
                label: l.kind ? `${l.label} (${l.kind})` : l.label,
                href: DRILL[name] ? (DRILL[name][tile.key(l)] ?? "") : "",
                total: Number(l.total),
                count: l.count,
                color: tile.color,
            }))
            // A zero line has no bar to draw; it stays in the table below.
            .filter((l) => l.total !== 0)
            .sort((a, b) => Math.abs(b.total) - Math.abs(a.total) || a.label.localeCompare(b.label));
    };

    // Open on the first tile that actually has something to draw, so the chart is
    // never a "nothing here" panel next to tiles full of money.
    const ORDER = ["income", "expense", "net_invested", "transfers_in", "uncategorized"];
    let selected = ORDER.find((name) => rowsFor(name).length) || "income";
    let rows = [];

    const rowH = 22;
    const padTop = 6;
    const padBottom = 16;
    const labelW = 130;
    const padRight = 8;

    // Labels are right-anchored at the label gutter's edge, so a long one
    // ("Investment redemption (redemption)") grows leftwards past x=0 and is
    // clipped by the viewBox. Trim it to what the gutter can hold, at ~0.54em per
    // character for this 10px font; the untruncated text stays in the bar's
    // aria-label and in the table below.
    const LABEL_MAX = Math.max(Math.floor((labelW - 6) / 5.4), 6);
    const clip = (s) => (s.length > LABEL_MAX ? s.slice(0, LABEL_MAX - 1).trimEnd() + "…" : s);

    // Room kept past the end of a bar for its amount text. Without it the peak bar
    // — which by definition spans its whole side — would run to the edge and its
    // label would be clipped on *every* chart, not just some edge case.
    const AMOUNT_GUTTER = 58;

    const draw = (width) => {
        // Height and scale follow the *selected* tile's lines, so they are
        // recomputed per draw rather than fixed once: a five-line bucket and a
        // one-line bucket are not the same chart.
        const height = padTop + padBottom + rows.length * rowH;
        const peak = Math.max(...rows.map((r) => Math.abs(r.total)), 0) || 1;
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.style.height = `${height}px`;
        const plotW = Math.max(width - labelW - padRight, 40);
        // The zero axis only needs to sit inboard when bars actually grow both
        // ways; a one-sided breakdown gives its side the whole plot.
        const hasPositive = rows.some((r) => r.total > 0);
        const hasNegative = rows.some((r) => r.total < 0);
        let zeroX;
        if (hasPositive && hasNegative) zeroX = labelW + plotW * 0.32;
        else if (hasNegative) zeroX = labelW + plotW;
        else zeroX = labelW;
        // Each side's bars are scaled to fit inside its gutter, so the longest
        // bar's label always lands on canvas. The floor keeps a very narrow chart
        // from producing a zero/negative bar length.
        const rightMax = Math.max(width - padRight - zeroX - AMOUNT_GUTTER, 10);
        const leftMax = Math.max(zeroX - labelW - AMOUNT_GUTTER, 10);
        const scale = (v) => (v >= 0 ? (v / peak) * rightMax : (v / peak) * leftMax);

        let body = "";
        rows.forEach((r, i) => {
            const y = padTop + i * rowH;
            const w = Math.abs(scale(r.total));
            const x = r.total >= 0 ? zeroX : zeroX - w;
            const barH = rowH - 7;
            const amount = fmtINR(r.total);
            // A contra line is dimmed rather than recolored: it belongs to its
            // bucket, it just points the other way.
            const opacity = r.total < 0 ? 0.45 : 0.85;
            const inner =
                `<rect x="0" y="${y.toFixed(1)}" width="${width}" height="${rowH}" fill="transparent" />` +
                `<text x="${labelW - 6}" y="${(y + barH - 1).toFixed(1)}" text-anchor="end" font-size="10" fill="currentColor" opacity="0.75">${esc(clip(r.label))}</text>` +
                `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(w, 1).toFixed(1)}" height="${barH}" fill="${r.color}" opacity="${opacity}" rx="2" />` +
                `<text x="${(r.total >= 0 ? x + w + 4 : x - 4).toFixed(1)}" y="${(y + barH - 1).toFixed(1)}" text-anchor="${r.total >= 0 ? "start" : "end"}" font-size="9" fill="currentColor" opacity="0.6">${esc(amount)}</text>`;
            const aria = `${esc(r.label)}: ${esc(amount)}, ${r.count} transactions`;
            // A bar with no rendered row to take its link from still gets drawn — a
            // silently missing bar would be money off the chart — it just is not
            // clickable.
            body += r.href
                ? `<a href="${esc(r.href)}" aria-label="${aria}">${inner}</a>`
                : `<g aria-label="${aria}">${inner}</g>`;
        });
        const axis = `<line x1="${zeroX.toFixed(1)}" x2="${zeroX.toFixed(1)}" y1="${padTop}" y2="${(height - padBottom).toFixed(1)}" stroke="currentColor" stroke-opacity="0.25" />`;
        svg.innerHTML = axis + body;
    };

    const render = sizedRenderer(svg, draw);

    const select = (name) => {
        selected = name;
        rows = rowsFor(name);
        tiles.forEach((tile) => {
            tile.setAttribute("aria-pressed", String(tile.dataset.select === name));
        });
        if (title) title.textContent = TILES[name].label;
        const nothing = !rows.length;
        empty.style.display = nothing ? "block" : "none";
        svg.style.display = nothing ? "none" : "block";
        caption.textContent = nothing
            ? ""
            : `${rows.length} lines · click a bar to list its transactions`;
        if (!nothing) render();
    };

    tiles.forEach((tile) => {
        // A click on the anchor *inside* a tile is that anchor's drill-through,
        // not a selection — let it navigate.
        tile.addEventListener("click", (event) => {
            if (event.target.closest("a")) return;
            select(tile.dataset.select);
        });
        // The tiles are role=button, so they owe the keyboard what a button gives
        // it — but a keydown on the anchor *inside* a tile bubbles up to here, so
        // the same guard the click handler uses is needed again: without it, Enter
        // on a focused drill-through link would be swallowed and select the tile
        // instead of following the link.
        tile.addEventListener("keydown", (event) => {
            if (event.target.closest("a")) return;
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            select(tile.dataset.select);
        });
    });

    select(selected);
    onResize(svg, render);
};

trend();
breakdown();
