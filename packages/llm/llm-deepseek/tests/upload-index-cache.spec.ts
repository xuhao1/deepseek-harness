import { mkdir, mkdtemp, readFile, rename, rm, stat, utimes, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AttachmentId, ImageVariantId } from '@deepseek-ai/dsh-attachment'
import { DeepSeekFileId } from '../src/file-id.ts'
import { deepSeekFileScope, DeepSeekUploadIndex } from '../src/upload-index.ts'
import type { DeepSeekUploadRecord } from '../src/upload-index.ts'

vi.mock('node:fs/promises', async (importOriginal) => {
  const fs = await importOriginal<typeof import('node:fs/promises')>()
  return { ...fs, readFile: vi.fn(fs.readFile), stat: vi.fn(fs.stat) }
})

const scope = deepSeekFileScope('https://api.deepseek.com', 'index-cache-test')
const roots: string[] = []

async function indexPath(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-upload-index-cache-'))
  roots.push(dir)
  return join(dir, 'index.json')
}

function record(id = 1): DeepSeekUploadRecord {
  return {
    scope,
    attachmentId: AttachmentId(`sha256:${'a'.repeat(64)}`),
    variantId: ImageVariantId(`sha256:${id.toString(16).padStart(64, '0')}`),
    fileId: DeepSeekFileId(`file-api-${id}`),
    bytes: 100,
    createdAt: 1,
    expiresAt: 10_000,
  }
}

async function seed(path: string, records: DeepSeekUploadRecord[]): Promise<void> {
  await writeFile(path, JSON.stringify({ formatVersion: 3, records }), 'utf8')
}

afterEach(async () => {
  vi.mocked(readFile).mockReset()
  vi.mocked(stat).mockReset()
  const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises')
  vi.mocked(readFile).mockImplementation(fs.readFile)
  vi.mocked(stat).mockImplementation(fs.stat)
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true })
})

describe('DeepSeekUploadIndex parsed cache', () => {
  it('reads an unchanged index once, rechecks expiry, and detaches returned records', async () => {
    const path = await indexPath()
    const entry = record()
    await seed(path, [entry])
    const index = new DeepSeekUploadIndex(path)

    const found = await index.get(scope, entry.variantId, 1, 1)
    expect(found).toEqual(entry)
    found!.fileId = DeepSeekFileId('caller-mutated')
    const rejected = await index.commit({ ...entry, fileId: DeepSeekFileId('duplicate') }, 1, 1)
    expect(rejected.accepted).toBe(false)
    rejected.record.fileId = DeepSeekFileId('caller-mutated-winner')
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
    await expect(index.get(scope, entry.variantId, 9_000, 1_000)).resolves.toBeUndefined()
    await expect(index.get(scope, entry.variantId, 9_000, 0)).resolves.toEqual(entry)
    await expect(index.get(scope, record(2).variantId, 1, 1)).resolves.toBeUndefined()
    expect(vi.mocked(readFile)).toHaveBeenCalledTimes(1)
  })

  it('reloads once after adding an image and retains exact invalidation and clear behavior', async () => {
    const path = await indexPath()
    const entries = [record(1), record(2), record(3)]
    await seed(path, entries)
    const index = new DeepSeekUploadIndex(path)
    for (const entry of entries) await index.get(scope, entry.variantId, 1, 1)
    expect(vi.mocked(readFile)).toHaveBeenCalledTimes(1)

    const added = record(4)
    await index.commit(added, 1, 1)
    entries.push(added)
    for (const entry of entries) await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
    expect(vi.mocked(readFile)).toHaveBeenCalledTimes(2)
    await index.remove(scope, [{ variantId: added.variantId, fileId: DeepSeekFileId('superseded') }])
    await expect(index.get(scope, added.variantId, 1, 1)).resolves.toEqual(added)
    expect(vi.mocked(readFile)).toHaveBeenCalledTimes(2)
    await index.remove(scope, [{ variantId: added.variantId, fileId: added.fileId }])
    await expect(index.get(scope, added.variantId, 1, 1)).resolves.toBeUndefined()
    await index.clear(scope)
    await expect(index.get(scope, entries[0]!.variantId, 1, 1)).resolves.toBeUndefined()
  })

  it('observes replacement by another index instance and preserves the other writer records', async () => {
    const path = await indexPath()
    const first = new DeepSeekUploadIndex(path)
    const second = new DeepSeekUploadIndex(path)
    const entry = record()
    await first.commit(entry, 1, 1)
    await expect(first.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
    const other = record(2)
    await second.commit(other, 1, 1)
    await expect(first.get(scope, other.variantId, 1, 1)).resolves.toEqual(other)
    const third = record(3)
    await first.commit(third, 1, 1)
    for (const candidate of [entry, other, third]) {
      await expect(second.get(scope, candidate.variantId, 1, 1)).resolves.toEqual(candidate)
    }
  })

  it('detects a same-size atomic replacement even when its modification time is restored', async () => {
    const path = await indexPath()
    const entry = record()
    await seed(path, [entry])
    const original = await stat(path)
    const index = new DeepSeekUploadIndex(path)
    await index.get(scope, entry.variantId, 1, 1)
    const replacement = { ...entry, fileId: DeepSeekFileId('file-api-2') }
    await seed(`${path}.new`, [replacement])
    await utimes(`${path}.new`, original.atime, original.mtime)
    await rename(`${path}.new`, path)
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(replacement)
  })

  it('does not retain bytes read across an external replacement', async () => {
    const path = await indexPath()
    const entry = record()
    const replacement = { ...entry, fileId: DeepSeekFileId('file-api-replacement') }
    await seed(path, [entry])
    const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises')
    vi.mocked(readFile).mockImplementationOnce(async (...args) => {
      const bytes = await fs.readFile(...args)
      await seed(`${path}.new`, [replacement])
      await rename(`${path}.new`, path)
      return bytes
    })
    const index = new DeepSeekUploadIndex(path)
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(replacement)
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(replacement)
    expect(vi.mocked(readFile)).toHaveBeenCalledTimes(2)
  })

  it('does not install an in-flight snapshot after a local write completes', async () => {
    const path = await indexPath()
    const entry = record()
    await seed(path, [entry])
    const index = new DeepSeekUploadIndex(path)
    const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises')
    vi.mocked(readFile).mockImplementationOnce(async (...args) => {
      const bytes = await fs.readFile(...args)
      await index.clear(scope)
      return bytes
    })
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toBeUndefined()
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toBeUndefined()
  })

  it('reloads after removal or corruption and repairs the durable index on commit', async () => {
    const path = await indexPath()
    const entry = record()
    await seed(path, [entry])
    const index = new DeepSeekUploadIndex(path)
    await index.get(scope, entry.variantId, 1, 1)
    await rm(path)
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toBeUndefined()
    await writeFile(path, '{bad', 'utf8')
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toBeUndefined()
    await expect(index.commit(entry, 1, 1)).resolves.toEqual({ accepted: true, record: entry })
    await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
  })

  it.each([0, 1])('retains only indexes within the serialized size ceiling (extra bytes: %i)', async (extra) => {
    const path = await indexPath()
    const entry = record()
    const text = JSON.stringify({ formatVersion: 3, records: [entry] })
    await writeFile(path, text.padEnd(8 * 1024 * 1024 + extra), 'utf8')
    const index = new DeepSeekUploadIndex(path)
    for (let i = 0; i < 2; i++) await expect(index.get(scope, entry.variantId, 1, 1)).resolves.toEqual(entry)
    expect(vi.mocked(readFile)).toHaveBeenCalledTimes(extra === 0 ? 1 : 2)
  })

  it('propagates filesystem failures after a previously cached read', async () => {
    const path = await indexPath()
    const entry = record()
    await seed(path, [entry])
    const index = new DeepSeekUploadIndex(path)
    await index.get(scope, entry.variantId, 1, 1)
    await rm(path)
    await mkdir(path)
    await expect(index.get(scope, entry.variantId, 1, 1)).rejects.toBeInstanceOf(Error)
    const denied = Object.assign(new Error('stat denied'), { code: 'EACCES' })
    vi.mocked(stat).mockRejectedValueOnce(denied)
    await expect(index.get(scope, entry.variantId, 1, 1)).rejects.toBe(denied)
  })
})
