import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

register("./cloudflare-loader.mjs", import.meta.url);

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${pathname}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("renders the MoveGuard unified RULA shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="ko">/i);
  assert.match(html, /MoveGuard 작업 자세 정밀평가 MVP/i);
  assert.match(html, /통합 RULA 측정/i);
  assert.match(html, /moveguard-mvp\.html\?embedded=1/i);
  assert.match(html, /분석 대시보드/i);
  assert.match(html, /원본 영상 저장 없음/i);
  assert.doesNotMatch(html, /codex-preview|Building your site/i);
});

test("ships live camera and upload modes through one RULA report", async () => {
  const html = await readFile(
    new URL("../dist/client/moveguard-mvp.html", import.meta.url),
    "utf8",
  );

  assert.match(html, /실시간 카메라 측정/i);
  assert.match(html, /촬영 영상 업로드/i);
  assert.match(html, /측정 종료·RULA 분석/i);
  assert.match(html, /같은 RULA 결과표/i);
  assert.match(html, /분석 방식/i);
  assert.match(html, /saveDashboardSession/i);
});

test("renders the analytics dashboard shell", async () => {
  const response = await render("/dashboard");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /측정 결과를 한눈에 확인하세요/i);
  assert.match(html, /새 측정 시작/i);
  assert.match(html, /측정 기록을 불러오고 있어요/i);
  assert.match(html, /수치만 저장/i);
});
