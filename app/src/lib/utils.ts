import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function downloadCsv(filename: string, headers: string[], rows: string[][]): void {
  const bom = "\uFEFF"
  const escapeField = (field: string) => {
    if (field.includes(",") || field.includes("\"") || field.includes("\n")) {
      return `"${field.replace(/"/g, "\"\"")}"`
    }
    return field
  }
  const csv = bom + [
    headers.map(escapeField).join(","),
    ...rows.map((row) => row.map(escapeField).join(",")),
  ].join("\n")
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
