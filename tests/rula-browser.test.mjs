import assert from "node:assert/strict";
import fs from "node:fs";

const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const scoring = html.slice(
  html.indexOf("function clamp("),
  html.indexOf("/* ---------- MediaPipe"),
);
const geometry = html.slice(
  html.indexOf("function angleBetween("),
  html.indexOf("function drawSkeleton("),
);

const loadFunctions = new Function(`
  ${scoring}
  ${geometry}
  return { neckScore, trunkScore, upperArmScore, lowerArmScore, wristScore, forceScore, computeFrameAngles };
`);
const rula = loadFunctions();

function neutralLandmarks() {
  const points = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  Object.assign(points[7], { x: 0.4, y: 0.3 });
  Object.assign(points[8], { x: 0.6, y: 0.3 });
  Object.assign(points[11], { x: 0.3, y: 0.5 });
  Object.assign(points[12], { x: 0.7, y: 0.5 });
  Object.assign(points[13], { x: 0.3, y: 0.7 });
  Object.assign(points[14], { x: 0.7, y: 0.7 });
  Object.assign(points[15], { x: 0.1, y: 0.7 });
  Object.assign(points[16], { x: 0.9, y: 0.7 });
  Object.assign(points[17], { x: 0.05, y: 0.7 });
  Object.assign(points[18], { x: 0.95, y: 0.7 });
  Object.assign(points[19], { x: 0.05, y: 0.7 });
  Object.assign(points[20], { x: 0.95, y: 0.7 });
  Object.assign(points[23], { x: 0.3, y: 0.9 });
  Object.assign(points[24], { x: 0.7, y: 0.9 });
  Object.assign(points[25], { x: 0.3, y: 0.94 });
  Object.assign(points[26], { x: 0.7, y: 0.94 });
  Object.assign(points[27], { x: 0.3, y: 0.97 });
  Object.assign(points[28], { x: 0.7, y: 0.97 });
  return points;
}

const landmarks = neutralLandmarks();
const angles = rula.computeFrameAngles(landmarks, landmarks);
assert.ok(Math.abs(angles.upperArmAngle) < 0.001, `neutral upper arm was ${angles.upperArmAngle}`);
assert.equal(rula.upperArmScore(angles.upperArmAngle), 1);
assert.equal(rula.neckScore(5, true), 2);
assert.equal(rula.neckScore(5, false, false, true), 4);
assert.equal(rula.trunkScore(10, false, true), 3);
assert.equal(rula.upperArmScore(10, true, true, false), 3);
assert.equal(rula.upperArmScore(10, true, true, true), 2);
assert.equal(rula.lowerArmScore(90, true), 2);
assert.equal(rula.wristScore(20, true), 4);
assert.equal(rula.forceScore("medium", false), 1);
assert.equal(rula.forceScore("medium", true), 2);

console.log("browser RULA regression tests passed");
