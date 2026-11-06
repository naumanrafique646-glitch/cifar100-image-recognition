const ALLOWED_ORIGIN = "https://naumanrafique646-glitch.github.io";
const DAILY_LIMIT = 60; // per-IP daily limit (saves + Gemini combined)
const PROMPT = "What is in this image? Answer in simple English: the object's name, its color, and 2-3 key details. Keep the answer short.";
const MIMES = ["image/jpeg", "image/png", "image/webp"];

const H = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, x-token",
  "Content-Type": "application/json",
};

async function limited(env, ip) {
  const day = new Date().toISOString().slice(0, 10);
  const k = "rl:" + ip + ":" + day;
  const n = parseInt((await env.MEMORY.get(k)) || "0", 10);
  if (n >= DAILY_LIMIT) return true;
  await env.MEMORY.put(k, String(n + 1), { expirationTtl: 90000 });
  return false;
}

async function handle(request, env) {
    const h = H;
    const json = (o, status = 200) => new Response(JSON.stringify(o), { status, headers: h });

    if (request.method === "OPTIONS") return new Response(null, { headers: h });
    const path = new URL(request.url).pathname;

    if (request.method === "GET") {
      const db = (await env.MEMORY.get("db")) || "";
      return new Response(db, { headers: { ...h, "Content-Type": "text/plain" } });
    }
    if (request.method !== "POST") return new Response("OK", { headers: h });

    // 1. Sirf tumhari site se
    if (request.headers.get("Origin") !== ALLOWED_ORIGIN)
      return json({ text: "Not authorized: site address mismatch.", ok: false }, 403);

    // 2. Token
    if (!env.APP_TOKEN || request.headers.get("x-token") !== env.APP_TOKEN)
      return json({ text: "Not authorized: token mismatch.", ok: false }, 401);

    // 3. Badi request band
    const len = parseInt(request.headers.get("content-length") || "0", 10);
    if (len > 3500000) return json({ text: "Image is too large.", ok: false }, 413);

    // 4. Daily limit
    const ip = request.headers.get("CF-Connecting-IP") || "unknown";
    let over = false;
    try { over = await limited(env, ip); } catch (e) { over = false; }
    if (over)
      return json({ text: "Daily limit reached. Please try again tomorrow.", ok: false }, 429);

    try {
      if (path === "/save") {
        const { e, t, h: hh } = await request.json();
        if (
          !Array.isArray(e) || e.length > 2048 || !e.every((v) => typeof v === "number") ||
          typeof t !== "string" || t.length > 3000
        ) return json({ ok: false }, 400);
        const old = (await env.MEMORY.get("db")) || "";
        if (old.length > 1500000) return json({ ok: false });
        const line = JSON.stringify({ e: e, h: String(hh || "").slice(0, 100), t: t });
        await env.MEMORY.put("db", old + line + "\n");
        return json({ ok: true });
      }

      const { image, mime } = await request.json();
      if (typeof image !== "string" || image.length > 3000000)
        return json({ text: "Invalid image.", ok: false }, 400);
      const mt = MIMES.includes(mime) ? mime : "image/jpeg";
      const body = JSON.stringify({
        contents: [{ parts: [{ text: PROMPT }, { inline_data: { mime_type: mt, data: image } }] }],
        generationConfig: { maxOutputTokens: 2048 },
      });

      const models = ["gemini-flash-lite-latest", "gemini-flash-latest"];
      let t = "Gemini is busy right now. Please try again shortly.";
      let ok = false;
      for (const m of models) {
        const r = await fetch(
          "https://generativelanguage.googleapis.com/v1beta/models/" + m + ":generateContent",
          { method: "POST", headers: { "Content-Type": "application/json", "x-goog-api-key": env.GEMINI_KEY }, body: body }
        );
        const d = await r.json();
        const x = (d.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("").trim();
        if (x) { t = x; ok = true; break; }
        if (d.error && d.error.code !== 503 && d.error.code !== 429) { t = JSON.stringify(d.error); break; }
      }
      return json({ text: t, ok: ok });
    } catch (e) {
      return json({ text: "Error: " + e.message, ok: false });
    }
}

export default {
  async fetch(request, env) {
    try {
      return await handle(request, env);
    } catch (e) {
      return new Response(JSON.stringify({ text: "Worker error: " + e.message, ok: false }), { status: 500, headers: H });
    }
  },
};
