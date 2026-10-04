/** Keyless benchmark for repeated lookups while the upload index grows. */

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { cpus, platform, release, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { performance } from 'node:perf_hooks'
import { pathToFileURL } from 'node:url'
import { AttachmentId, ImageVariantId } from '@deepseek-ai/dsh-attachment'
import { DeepSeekFileId } from '../src/file-id.ts'
import type { DeepSeekUploadRecord } from '../src/upload-index.ts'

const moduleUrl = process.argv[2] === undefined
  ? new URL('../src/upload-index.ts', import.meta.url).href
  : pathToFileURL(resolve(process.argv[2])).href
const { DeepSeekUploadIndex, deepSeekFileScope } = await import(moduleUrl) as typeof import('../src/upload-index.ts')
const scope = deepSeekFileScope('https://example.invalid', 'benchmark-only')
const now = 1_000
const warmups = 2
const samples = 5

function record(number: number): DeepSeekUploadRecord {
  const digest = createHash('sha256').update(String(number)).digest('hex')
  return {
    scope,
    attachmentId: AttachmentId(`sha256:${digest}`),
    variantId: ImageVariantId(`sha256:${digest}`),
    fileId: DeepSeekFileId(`file-api-benchmark-${number}`),
    bytes: 100_000,
    createdAt: now,
    expiresAt: now + 3_600_000,
  }
}

const root = await mkdtemp(join(tmpdir(), 'dsh-upload-index-benchmark-'))
try {
  const results = []
  for (const initialRecords of [100, 500]) {
    const path = join(root, `${initialRecords}.json`)
    const records = Array.from({ length: initialRecords }, (_, number) => record(number))
    await writeFile(path, JSON.stringify({ formatVersion: 3, records }), 'utf8')
    const index = new DeepSeekUploadIndex(path)
    const elapsedMs = []
    const lookupCounts = []
    for (let batch = 0; batch < warmups + samples; batch++) {
      const next = record(records.length)
      await index.commit(next, now, 60_000)
      records.push(next)
      const started = performance.now()
      for (const expected of records) {
        assert.equal((await index.get(scope, expected.variantId, now, 60_000))?.fileId, expected.fileId)
      }
      const elapsed = performance.now() - started
      if (batch >= warmups) {
        elapsedMs.push(elapsed)
        lookupCounts.push(records.length)
      }
    }
    const sorted = [...elapsedMs].sort((a, b) => a - b)
    results.push({ initialRecords, lookupCounts, elapsedMs, medianMs: sorted[Math.floor(sorted.length / 2)] })
  }
  console.log(JSON.stringify({
    node: process.version,
    platform: platform(),
    kernel: release(),
    cpu: cpus()[0]?.model,
    warmups,
    samples,
    scope: 'Lookup batches only; a new mapping is committed before each batch. No network or model requests.',
    results,
  }, null, 2))
} finally {
  await rm(root, { recursive: true, force: true })
}
