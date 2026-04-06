import type { UploadItem } from "@/lib/types"

type UploadMode = "image" | "folder"

type DataTransferItemWithEntry = DataTransferItem & {
  webkitGetAsEntry?: () => FileSystemEntry | null
}

const IMAGE_FILE_PATTERN = /\.(avif|bmp|gif|jpe?g|png|tiff?|webp)$/i

function isImageFile(file: File): boolean {
  return file.type.startsWith("image/") || IMAGE_FILE_PATTERN.test(file.name)
}

function normalizeRelativePath(rawPath: string): string {
  return rawPath.replace(/^\/+/, "").replaceAll("\\", "/")
}

function relativePathFromFile(file: File): string {
  if ("webkitRelativePath" in file && typeof file.webkitRelativePath === "string") {
    return normalizeRelativePath(file.webkitRelativePath || file.name)
  }
  return file.name
}

function finalizeItems(items: UploadItem[], mode: UploadMode): UploadItem[] {
  const nextItems = items
    .filter((item) => isImageFile(item.file))
    .sort((left, right) => left.relativePath.localeCompare(right.relativePath, "zh-CN"))

  return mode === "image" ? nextItems.slice(0, 1) : nextItems
}

export function filesToUploadItems(files: Iterable<File>, mode: UploadMode): UploadItem[] {
  return finalizeItems(
    Array.from(files).map((file) => ({
      file,
      relativePath: relativePathFromFile(file),
    })),
    mode
  )
}

function readFileEntry(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => {
    entry.file(resolve, reject)
  })
}

function readDirectoryBatch(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => {
    reader.readEntries(resolve, reject)
  })
}

async function readDirectoryEntries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  const entries: FileSystemEntry[] = []
  while (true) {
    const batch = await readDirectoryBatch(reader)
    if (!batch.length) {
      break
    }
    entries.push(...batch)
  }
  return entries
}

async function walkEntry(
  entry: FileSystemEntry,
  prefix = ""
): Promise<UploadItem[]> {
  if (entry.isFile) {
    const file = await readFileEntry(entry as FileSystemFileEntry)
    if (!isImageFile(file)) {
      return []
    }
    return [
      {
        file,
        relativePath: normalizeRelativePath(prefix ? `${prefix}/${file.name}` : file.name),
      },
    ]
  }

  if (!entry.isDirectory) {
    return []
  }

  const basePath = prefix ? `${prefix}/${entry.name}` : entry.name
  const directoryReader = (entry as FileSystemDirectoryEntry).createReader()
  const children = await readDirectoryEntries(directoryReader)
  const nested = await Promise.all(children.map((child) => walkEntry(child, basePath)))
  return nested.flat()
}

export async function readDroppedItems(
  dataTransfer: DataTransfer,
  mode: UploadMode
): Promise<UploadItem[]> {
  const entries = Array.from(dataTransfer.items ?? [])
    .map((item) => (item as DataTransferItemWithEntry).webkitGetAsEntry?.() ?? null)
    .filter((entry): entry is FileSystemEntry => entry !== null)

  if (entries.length) {
    const nested = await Promise.all(entries.map((entry) => walkEntry(entry)))
    return finalizeItems(nested.flat(), mode)
  }

  return filesToUploadItems(Array.from(dataTransfer.files), mode)
}
