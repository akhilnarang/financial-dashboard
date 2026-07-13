// Shared machinery for the site's hand-rolled inline-SVG charts.
//
// Every chart here sizes itself to its container's real pixel width (1 viewBox
// unit = 1 CSS px) and rebuilds its innerHTML on each draw, so text stays crisp
// without preserveAspectRatio tricks — and anything that reads a node out of the
// SVG has to re-query it after a draw rather than hold a reference.
//
// Monetary fields arrive as JSON *strings* (they are Decimals on the server, and
// serializing them as floats would be a rounding decision made in the wrong
// place), so every one of them must be coerced with Number() before arithmetic.

const MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** Compact rupee text: ₹1.25Cr / ₹3.40L / ₹9.9K / ₹842. */
export const fmtINR = (v) => {
    const n = Math.round(Number(v));
    if (Math.abs(n) >= 1e7) return "₹" + (n / 1e7).toFixed(2) + "Cr";
    if (Math.abs(n) >= 1e5) return "₹" + (n / 1e5).toFixed(2) + "L";
    if (Math.abs(n) >= 1e3) return "₹" + (n / 1e3).toFixed(1) + "K";
    return "₹" + n.toLocaleString("en-IN");
};

/** "2026-07" -> "Jul 2026". */
export const fmtMonth = (ym) => {
    const [y, m] = ym.split("-");
    return `${MONTH_NAMES[parseInt(m, 10) - 1]} ${y}`;
};

/**
 * HTML-escape a value being interpolated into an SVG string.
 *
 * The charts build their markup as text and assign it to innerHTML, so every
 * value out of the ledger — a category slug, a counterparty, an href — passes
 * through here first. Nothing may skip it.
 */
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

/** Read and parse a `<script type="application/json">` payload; {} if absent. */
export const jsonBlock = (id, fallback = {}) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    try {
        return JSON.parse(el.textContent);
    } catch (error) {
        console.debug("Malformed JSON payload", id, error);
        return fallback;
    }
};

// ~½s of frames at 60fps. Past that, a still-unsized SVG is a hidden or detached
// one, and the ResizeObserver below will pick it up if it ever gets a size.
const RETRY_LIMIT = 30;

/**
 * Wrap `draw(width)` in the deferral every chart needs before its first paint.
 *
 * A hidden or not-yet-laid-out parent reports a width of 0. Drawing at a
 * fallback width and then snapping to the real one is visible flicker, so the
 * first paint waits for a real width instead — but only for a bounded number of
 * frames, or a permanently hidden SVG would spin requestAnimationFrame forever.
 */
export const sizedRenderer = (svg, draw, minWidth = 320) => {
    let retries = 0;
    const render = () => {
        const raw = svg.clientWidth || svg.getBoundingClientRect().width;
        if (!raw) {
            if (!svg.isConnected) return;
            if (retries++ < RETRY_LIMIT) requestAnimationFrame(render);
            return;
        }
        retries = 0;
        draw(Math.max(raw, minWidth));
    };
    return render;
};

/**
 * Re-render, debounced, whenever the container's width changes (window resize,
 * sidebar toggle). The draw rebuilds innerHTML; listeners survive because the
 * hover code re-queries its nodes on every use.
 */
export const onResize = (svg, render) => {
    if (typeof ResizeObserver === "undefined") return;
    let timer = null;
    new ResizeObserver(() => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(render, 100);
    }).observe(svg);
};

/**
 * Convert client (screen) coordinates into the SVG's viewBox space, or null.
 *
 * getScreenCTM is invertible for any rendered SVG in practice, but inverse()
 * throws on a degenerate matrix (a parent going display:none mid-render), and a
 * hover is never worth an exception.
 */
export const toViewBox = (svg, clientX, clientY) => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    let inverse;
    try {
        inverse = ctm.inverse();
    } catch (_) {
        return null;
    }
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY ?? 0;
    const mapped = pt.matrixTransform(inverse);
    return Number.isFinite(mapped.x) ? mapped : null;
};

/** Index of the item whose `centreOf(item)` x is nearest `x`. -1 when empty. */
export const nearestByX = (items, x, centreOf) => {
    let nearest = -1;
    let best = Infinity;
    for (let i = 0; i < items.length; i++) {
        const d = Math.abs(centreOf(items[i]) - x);
        if (d < best) {
            best = d;
            nearest = i;
        }
    }
    return nearest;
};

const LABEL_H = 14;

/**
 * Position a tooltip pill (a <rect> behind a centred <text>) at an anchor point.
 *
 * The width is the text's *measured* length, which beats a per-character
 * estimate for a proportional font; getComputedTextLength can throw before
 * layout, so a rough proxy backs it up. The pill is then clamped inside the
 * plot — capping its width at the drawable area first, so the clamp can never
 * be handed a min above its max on a very narrow chart — and flipped below the
 * anchor when there is no room for it above.
 */
export const placeTooltip = (
    { label, bg, text, anchorX, anchorY, leftEdge, rightEdge, drawableW, padTop, aboveGap, belowGap },
) => {
    label.textContent = text;
    let labelW;
    try {
        labelW = label.getComputedTextLength() + 12;
    } catch (_) {
        labelW = text.length * 6 + 12;
    }
    labelW = Math.min(labelW, Math.max(drawableW, 20));
    const halfW = labelW / 2;

    let labelY = anchorY - aboveGap;
    if (labelY - LABEL_H < padTop) labelY = anchorY + LABEL_H + belowGap;
    const labelX = Math.min(Math.max(anchorX, leftEdge + halfW), rightEdge - halfW);

    label.setAttribute("x", labelX.toFixed(1));
    label.setAttribute("y", labelY.toFixed(1));
    bg.setAttribute("x", (labelX - halfW).toFixed(1));
    bg.setAttribute("y", (labelY - LABEL_H + 3).toFixed(1));
    bg.setAttribute("width", labelW.toFixed(1));
    bg.setAttribute("height", LABEL_H.toFixed(1));
};

/**
 * Bind hover-to-inspect for both pointer kinds.
 *
 * touchend/touchcancel are not optional: without them the pill sticks on iOS
 * after the finger lifts.
 */
export const bindHover = (svg, show, hide) => {
    svg.addEventListener("mousemove", (e) => show(e.clientX, e.clientY));
    svg.addEventListener("mouseleave", hide);
    const touch = (e) => {
        if (e.touches[0]) show(e.touches[0].clientX, e.touches[0].clientY);
    };
    svg.addEventListener("touchstart", touch, { passive: true });
    svg.addEventListener("touchmove", touch, { passive: true });
    svg.addEventListener("touchend", hide, { passive: true });
    svg.addEventListener("touchcancel", hide, { passive: true });
};

/** Markup for the hover pill: a rounded background under a centred label. */
export const tooltipMarkup = (id) =>
    `<rect id="${id}-bg" x="0" y="0" width="0" height="0" fill="currentColor" opacity="0.92" rx="3" />` +
    `<text id="${id}-label" x="0" y="0" font-size="10" font-weight="600" text-anchor="middle" fill="var(--clr-bg, #000)"></text>`;
