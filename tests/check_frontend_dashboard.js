// Two dashboard layout rules that were broken, checked against the stylesheet.
//
// Neither is visible in a unit test of behaviour: they are CSS interactions, and the
// way they failed was geometric. Both were measured in a real browser first.
//
//  1. #dashgrid carried `perspective:1600px`, added for the panel tilt. A perspective
//     other than `none` makes an element the containing block for any fixed-position
//     descendant, so an expanded panel resolved `position:fixed` against the
//     scrolling grid rather than the window. Measured: it landed at top 368 instead
//     of 56, and was 356px tall where it should have been 668 -- exactly the grid's
//     box minus the insets. The perspective now lives in each panel's own transform.
//
//  2. The row track collapsed to an equal share of the visible height: 180px a row
//     for panels whose content was 416px, so every panel was clipped to about half
//     of itself. grid-auto-rows:max-content lets rows take what they need and the
//     grid scroll.
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const css = (html.match(/<style>([\s\S]*?)<\/style>/) || [])[1] || "";

// Every declaration written for one selector, with comments stripped.
//
// Anchored at a rule boundary, and collecting all matches rather than the first,
// because neither shortcut survives this stylesheet: ".panel-body" also appears at
// the tail of ".panel.zoom .panel-body", and "#dashgrid>.panel" is written more than
// once. Matching the first substring read the wrong rule both times and called a
// correct stylesheet broken.
function rule(selector) {
  const clean = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const esc = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp("(?:^|[}\\n;])\\s*" + esc + "\\s*\\{([^}]*)\\}", "g");
  const found = [];
  let m;
  while ((m = re.exec(clean)) !== null) found.push(m[1]);
  return found.length ? found.join(";").replace(/\s+/g, " ").trim() : null;
}

let bad = 0;
const check = (name, ok, detail) => {
  console.log((ok ? "PASS  " : "FAIL  ") + name + (detail ? `   ${detail}` : ""));
  if (!ok) bad++;
};

const grid = rule("#dashgrid");
check("the dashboard grid exists", !!grid);

// The bug itself: a perspective here re-parents every fixed-position descendant.
check("the grid creates no containing block for fixed children",
      !!grid && !/\bperspective\s*:\s*(?!none)/.test(grid),
      grid && /perspective[^;]*/.exec(grid) ? RegExp.lastMatch : "no perspective");
check("...and no transform either, which would do the same",
      !!grid && !/\btransform\s*:\s*(?!none)/.test(grid));
check("...and no filter, which would do the same again",
      !!grid && !/\bfilter\s*:\s*(?!none)/.test(grid));

check("rows take the height their content needs",
      !!grid && /grid-auto-rows\s*:\s*(max-content|min-content|auto)/.test(grid),
      grid && /grid-auto-rows[^;]*/.exec(grid) ? RegExp.lastMatch : "not set");
check("the grid scrolls rather than squeezing",
      !!grid && /overflow-y\s*:\s*auto/.test(grid));

// The tilt still has to be a real rotation, so the perspective moved rather than left.
const panel = rule("#dashgrid>.panel");
check("panels still tilt in a perspective of their own",
      !!panel && /transform\s*:[^;]*perspective\(/.test(panel),
      panel ? (/transform[^;]*/.exec(panel) || [""])[0].slice(0, 62) : "no rule");

const zoom = rule(".panel.zoom");
check("an expanded panel is fixed to the window", !!zoom && /position\s*:\s*fixed/.test(zoom));
check("...and clears any tilt it was carrying",
      !!zoom && /transform\s*:\s*none\s*!important/.test(zoom));
check("...and is inset from the window edges", !!zoom && /inset\s*:/.test(zoom),
      zoom ? (/inset[^;]*/.exec(zoom) || [""])[0] : "");

check("its body is allowed the full height",
      (rule(".panel.zoom .panel-body") || "").includes("max-height:none"));

// A percentage height cannot resolve against a parent sized by the flex layout, so
// the SVG fell back to intrinsic sizing: measured at 2.03x magnification with both
// axes pushed past the bottom edge of the panel. A definite height lets
// preserveAspectRatio="meet" fit the whole drawing inside and letterbox the sides.
const zoomChart = rule(".panel.zoom .chart") || "";
check("an expanded chart is given a real height, not a percentage",
      /height\s*:/.test(zoomChart) && !/height\s*:\s*[\d.]+%/.test(zoomChart),
      (/height[^;]*/.exec(zoomChart) || ["not set"])[0]);
check("...bounded so the drawing cannot outgrow the panel",
      /height *:[^;]*(vh|px|rem|em)/.test(zoomChart), zoomChart.slice(0, 54));
check("the way out stays on screen without hovering",
      /\.panel\.zoom \.panel-tools\s*\{[^}]*opacity\s*:\s*1/.test(
          css.replace(/\/\*[\s\S]*?\*\//g, "")));

// A panel that is not expanded still needs a cap, or one long table pushes the rest
// of the dashboard off the screen.
const body = rule(".panel-body");
check("an ordinary panel body is still capped",
      !!body && /max-height\s*:\s*min\(/.test(body),
      body ? (/max-height[^;]*/.exec(body) || [""])[0] : "");

console.log(bad ? `\n${bad} PROBLEM(S)` : "\nPASS  the dashboard lays out and expands correctly");
process.exit(bad ? 1 : 0);
