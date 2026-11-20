// Open-model load test: a fixed arrival rate regardless of response times, so slow
// responses show up as latency instead of silently lowering the request rate.
import http from "k6/http";
import { check } from "k6";
import { Counter, Trend } from "k6/metrics";

const BASE = __ENV.BASE_URL || "http://api:8000";
const QPS = Number(__ENV.QPS || 50);
const users = JSON.parse(open(__ENV.USERS || "/out/users.json"));

export const options = {
  discardResponseBodies: false,
  summaryTrendStats: ["avg", "p(50)", "p(95)", "p(99)", "max"],
  scenarios: {
    load: {
      executor: "constant-arrival-rate",
      rate: QPS,
      timeUnit: "1s",
      duration: __ENV.DURATION || "60s",
      preAllocatedVUs: Math.max(20, QPS),
      maxVUs: Math.max(100, QPS * 4),
    },
  },
  thresholds: __ENV.SMOKE ? { http_req_failed: ["rate<0.01"], http_req_duration: ["p(99)<500"] } : {},
};

const stages = ["features", "retrieve", "rank", "rerank", "total"];
const trends = Object.fromEntries(stages.map((s) => [s, new Trend(`server_${s}`, true)]));
const fallbacks = new Counter("fallbacks");

export default function () {
  const uid = users[Math.floor(Math.random() * users.length)];
  const res = http.get(`${BASE}/recommend?user_id=${uid}&k=10`);
  const ok = check(res, { "status 200": (r) => r.status === 200 });
  if (!ok) return;
  const body = res.json();
  if (body.fallback) fallbacks.add(1);
  for (const s of stages) {
    if (body.timings_ms[s] !== undefined) trends[s].add(body.timings_ms[s]);
  }
}

export function handleSummary(data) {
  return { [__ENV.OUT || "/out/summary.json"]: JSON.stringify(data) };
}
