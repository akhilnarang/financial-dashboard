// The net-worth page's trend: one line, area-filled, over a month-end series.
//
// The series is fetched rather than serialized into the page, because unlike the
// figures beside it the trend is not something the page was rendered from — it is
// its own read across every month of snapshot history.

import {
    bindHover,
    fmtINR,
    fmtMonth,
    nearestByX,
    onResize,
    placeTooltip,
    sizedRenderer,
    toViewBox,
    tooltipMarkup,
} from "./charts.js";

const trend = async () => {
    const svg = document.getElementById("nw-trend");
    const caption = document.getElementById("nw-trend-caption");
    const empty = document.getElementById("nw-trend-empty");
    if (!svg) return;

    let response;
    try {
        response = await fetch(svg.dataset.trendUrl, { cache: "no-store" });
    } catch (error) {
        console.debug("Failed to load net-worth trend", error);
        return;
    }
    if (!response.ok) return;
    const points = await response.json();
    // One point is not a trend: a single month has nothing to draw a line between.
    if (!points || points.length < 2) {
        if (empty) empty.style.display = "block";
        return;
    }

    const values = points.map((p) => Number(p.value));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    // Closure state written by draw() and read by showHover().
    let coords = [];
    let width = 600;
    const padX = 36;
    const height = 140;
    const padY = 20;
    const padTop = 8;

    const draw = (rawWidth) => {
        width = rawWidth;
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        const span = points.length - 1;
        coords = points.map((p, i) => {
            const x = padX + (i * (width - padX - 8)) / span;
            const y = padTop + (height - padTop - padY) * (1 - (Number(p.value) - min) / range);
            return [x, y];
        });
        const linePath = coords.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
        const baselineY = (height - padY).toFixed(1);
        const areaPath = `M${coords[0][0].toFixed(1)},${baselineY} L${linePath} L${coords[coords.length - 1][0].toFixed(1)},${baselineY} Z`;
        const last = coords[coords.length - 1];

        const yLabels = [
            { v: max, y: padTop },
            { v: (max + min) / 2, y: padTop + (height - padTop - padY) / 2 },
            { v: min, y: height - padY },
        ];
        const yTicksSvg = yLabels.map((t) =>
            `<text x="${padX - 4}" y="${(t.y + 3).toFixed(1)}" text-anchor="end" font-size="9" fill="currentColor" opacity="0.55">${fmtINR(t.v)}</text>` +
            `<line x1="${padX}" x2="${width - 8}" y1="${t.y.toFixed(1)}" y2="${t.y.toFixed(1)}" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="2 3" />`
        ).join("");
        const xLabelsSvg =
            `<text x="${padX.toFixed(1)}" y="${height - 4}" text-anchor="start" font-size="9" fill="currentColor" opacity="0.55">${fmtMonth(points[0].month)}</text>` +
            `<text x="${(width - 8).toFixed(1)}" y="${height - 4}" text-anchor="end" font-size="9" fill="currentColor" opacity="0.55">${fmtMonth(points[points.length - 1].month)}</text>`;

        const dotsSvg = coords.map(([x, y]) =>
            `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.5" fill="currentColor" opacity="0.4" />`
        ).join("");

        const hoverSvg =
            `<g id="nw-hover" style="opacity: 0; pointer-events: none; transition: opacity 0.1s;">` +
            `<line id="nw-hover-line" x1="0" x2="0" y1="${padTop}" y2="${height - padY}" stroke="currentColor" stroke-opacity="0.3" stroke-width="1" stroke-dasharray="3 3" />` +
            `<circle id="nw-hover-dot" cx="0" cy="0" r="4" fill="currentColor" />` +
            tooltipMarkup("nw-hover") +
            `</g>`;

        svg.innerHTML =
            yTicksSvg +
            `<path d="${areaPath}" fill="currentColor" opacity="0.08" />` +
            `<polyline points="${linePath}" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.9" />` +
            dotsSvg +
            `<circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="3" fill="currentColor" />` +
            xLabelsSvg +
            hoverSvg;
    };

    const render = sizedRenderer(svg, draw);
    render();

    const first = values[0];
    const lastV = values[values.length - 1];
    const delta = lastV - first;
    const deltaStr = (delta >= 0 ? "+" : "−") + fmtINR(Math.abs(delta));
    caption.innerHTML =
        `<span>${fmtMonth(points[0].month)}: <strong>${fmtINR(first)}</strong> &rarr; ${fmtMonth(points[points.length - 1].month)}: <strong>${fmtINR(lastV)}</strong></span>` +
        `<span>net change: <strong>${deltaStr}</strong> &middot; ${points.length} months &middot; <span class="text-sm text-muted">hover to inspect</span></span>`;

    const showHover = (clientX, clientY) => {
        if (!coords.length) return;   // draw() bailed before laying out points
        const point = toViewBox(svg, clientX, clientY);
        if (!point) return;
        const nearest = nearestByX(coords, point.x, ([x]) => x);
        if (nearest < 0) return;
        const [px, py] = coords[nearest];

        // Re-queried, not held: every draw rebuilds the SVG's innerHTML.
        const hoverGroup = svg.querySelector("#nw-hover");
        if (!hoverGroup) return;
        const hoverLine = svg.querySelector("#nw-hover-line");
        const hoverDot = svg.querySelector("#nw-hover-dot");

        hoverLine.setAttribute("x1", px.toFixed(1));
        hoverLine.setAttribute("x2", px.toFixed(1));
        hoverDot.setAttribute("cx", px.toFixed(1));
        hoverDot.setAttribute("cy", py.toFixed(1));

        placeTooltip({
            label: svg.querySelector("#nw-hover-label"),
            bg: svg.querySelector("#nw-hover-bg"),
            text: `${fmtMonth(points[nearest].month)} · ${fmtINR(values[nearest])}`,
            anchorX: px,
            anchorY: py,
            leftEdge: padX,
            rightEdge: width - 4,
            drawableW: Math.max(width - padX - 8, 20),
            padTop,
            aboveGap: 10,
            belowGap: 4,
        });
        hoverGroup.style.opacity = "1";
    };

    const hideHover = () => {
        const hg = svg.querySelector("#nw-hover");
        if (hg) hg.style.opacity = "0";
    };

    bindHover(svg, showHover, hideHover);
    onResize(svg, render);
};

trend();
