import { GlobalWorkerOptions } from "pdfjs-dist/legacy/build/pdf.worker.entry";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();
