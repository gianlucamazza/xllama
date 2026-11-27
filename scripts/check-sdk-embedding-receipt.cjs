// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Replay captured Xbox embedding requests through the official JavaScript SDK.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const OpenAI = require("openai").default;
const { VERSION } = require("openai/version");

async function checkRow(row) {
  const payload = { ...row.request };
  const encoding = payload.encoding_format;
  // Exercise the SDK default: base64 on the wire, floats for the caller.
  if (encoding === "base64") delete payload.encoding_format;
  const client = new OpenAI({
    apiKey: "offline",
    baseURL: "http://receipt.invalid/v1",
    maxRetries: 0,
    fetch: async (url, init) => {
      assert.equal(init.method, "POST");
      assert.equal(new URL(url).pathname, row.route);
      assert.deepEqual(JSON.parse(init.body), row.request);
      return Response.json(row.response, { status: row.status });
    },
  });
  const response = await client.embeddings.create(payload);
  assert.equal(response.object, "list");
  assert.equal(response.model, row.response.model);
  assert.deepEqual(response.usage, row.response.usage);
  assert.ok(response.data.length);
  assert.equal(response.data.length, row.response.data.length);
  for (const [i, item] of response.data.entries()) {
    const raw = row.response.data[i];
    assert.equal(item.index, raw.index);
    let vector = raw.embedding;
    if (encoding === "base64") {
      const bytes = Buffer.from(vector, "base64");
      assert.equal(bytes.length % 4, 0);
      vector = Array.from({ length: bytes.length / 4 }, (_, n) =>
        bytes.readFloatLE(n * 4),
      );
    }
    assert.ok(vector.length);
    assert.deepEqual(Array.from(item.embedding), vector);
  }
}

async function main() {
  const receipt = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const checked = new Set();
  for (const row of receipt.requests) {
    if (
      row.route === "/v1/embeddings" &&
      row.status === 200 &&
      ["float", "base64"].includes(row.request.encoding_format)
    ) {
      await checkRow(row);
      checked.add(row.request.encoding_format);
    }
  }
  assert.equal(
    checked.size,
    2,
    "receipt requires successful float and base64 responses",
  );
  console.log(`PASS openai-node ${VERSION}: float and default base64`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
