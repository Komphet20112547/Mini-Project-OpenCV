"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type EmotionResult = {
  label: string;
  confidence: number;
};

// คำแปลอารมณ์เป็นภาษาไทย
const emotionTranslations: Record<string, string> = {
  angry: "โกรธ",
  disgust: "รังเกียจ",
  fear: "กลัว",
  happy: "มีความสุข",
  neutral: "เฉยๆ",
  sad: "เศร้า",
  surprise: "ประหลาดใจ",
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<string>(
    "พร้อม เริ่มกดปุ่ม Start",
  );
  const [emotion, setEmotion] = useState<EmotionResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const processingIntervalRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const isInferBusyRef = useRef(false);
  const lastInferTsRef = useRef(0);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const labelsRef = useRef<string[] | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const cascadeRef = useRef<any | null>(null);
  const cvMatsRef = useRef<{
    src?: any;
    gray?: any;
    rgb?: any;
    face?: any;
    resized?: any;
    faces?: any;
    size0?: any;
    size64?: any;
  }>({});
  const floatInputRef = useRef<Float32Array | null>(null);

  // Load OpenCV script dynamically
  const loadOpenCV = useCallback(async () => {
    if (typeof window === "undefined") return;
    if ((window as any).cv) {
      return new Promise<void>((resolve) => {
        const cv = (window as any).cv;
        if (cv && cv.ready) {
          resolve();
        } else {
          cv["onRuntimeInitialized"] = () => {
            resolve();
          };
        }
      });
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) {
          reject(new Error("ไม่สามารถโหลด OpenCV ได้"));
          return;
        }
        cv["onRuntimeInitialized"] = () => {
          resolve();
        };
      };
      script.onerror = () =>
        reject(new Error("โหลดสคริปต์ OpenCV ล้มเหลว"));
      document.body.appendChild(script);
    });
  }, []);

  // Load ONNX model and labels
  const loadModel = useCallback(async () => {
    if (sessionRef.current && labelsRef.current) return;

    const [labelsRes] = await Promise.all([
      fetch("/models/classes.json"),
    ]);

    if (!labelsRes.ok) {
      throw new Error("โหลด classes.json ไม่สำเร็จ");
    }

    const labels = (await labelsRes.json()) as string[];
    labelsRef.current = labels;

    // Deterministic WASM asset loading (we'll copy wasm files into /public/ort in postinstall)
    ort.env.wasm.wasmPaths = "/ort/";
    // Keep it stable across devices; threaded wasm can be flaky depending on COOP/COEP headers
    ort.env.wasm.numThreads = 1;

    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      {
        executionProviders: ["wasm"],
      },
    );
    sessionRef.current = session;
    const firstInputName = session.inputNames[0];
    const inputMeta = firstInputName
      ? (session.inputMetadata as Record<string, any>)[firstInputName]
      : null;
    console.log("โมเดลโหลดสำเร็จ:", {
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      inputShape: inputMeta?.dims ?? null,
    });
  }, []);

  // Load Haar cascade for face detection
  const loadCascade = useCallback(async () => {
    if (cascadeRef.current) return cascadeRef.current;
    const cv = (window as any).cv;
    if (!cv) return null;

    const response = await fetch(
      "/opencv/haarcascade_frontalface_default.xml",
    );
    if (!response.ok) {
      throw new Error("โหลด Haar cascade ไม่สำเร็จ");
    }
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    const fileName = "haarcascade_frontalface_default.xml";
    // Avoid re-creating the same file in OpenCV FS
    const existing = cv.FS_analyzePath?.(`/${fileName}`)?.exists;
    if (!existing) {
      cv.FS_createDataFile(
        "/", // path
        fileName,
        data,
        true,
        false,
      );
    }
    const classifier = new cv.CascadeClassifier();
    classifier.load(fileName);
    cascadeRef.current = classifier;
    return cascadeRef.current;
  }, []);

  const stopProcessing = useCallback(() => {
    if (processingIntervalRef.current) {
      window.clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    isInferBusyRef.current = false;
  }, []);

  const stopCamera = useCallback(() => {
    stopProcessing();
    const video = videoRef.current;
    if (video && video.srcObject) {
      const stream = video.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    // cleanup OpenCV mats
    const mats = cvMatsRef.current;
    Object.values(mats).forEach((m) => {
      try {
        m?.delete?.();
      } catch {
        // ignore
      }
    });
    cvMatsRef.current = {};
    setIsRunning(false);
  }, [stopProcessing]);

  const softmaxTop1 = useCallback(
    (scores: Float32Array) => {
      let max = -Infinity;
      for (let i = 0; i < scores.length; i++) max = Math.max(max, scores[i]);
      let sum = 0;
      const exps = new Float32Array(scores.length);
      for (let i = 0; i < scores.length; i++) {
        const e = Math.exp(scores[i] - max);
        exps[i] = e;
        sum += e;
      }
      let bestIdx = 0;
      let bestP = exps[0] / sum;
      for (let i = 1; i < exps.length; i++) {
        const p = exps[i] / sum;
        if (p > bestP) {
          bestP = p;
          bestIdx = i;
        }
      }
      return { bestIdx, bestP };
    },
    [],
  );

  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true);
      setErrorMessage(null);
      setStatus("กำลังเตรียมโมเดลและกล้อง...");

      await loadOpenCV();
      await loadModel();
      const cascade = await loadCascade();
      if (!cascade) {
        throw new Error("ไม่สามารถสร้างตัวตรวจจับใบหน้าได้");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      const video = videoRef.current;
      if (!video) throw new Error("ไม่พบ element ของวิดีโอ");

      video.srcObject = stream;
      await video.play();

      setIsRunning(true);
      setStatus("กำลังทำงาน... แสดงใบหน้าของคุณต่อกล้อง");

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const cv = (window as any).cv;

      // Fix canvas size once after video is ready
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Reuse mats/buffers to reduce GC + keep FPS stable
      const mats = cvMatsRef.current;
      mats.src = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
      mats.gray = new cv.Mat();
      mats.rgb = new cv.Mat();
      mats.face = new cv.Mat();
      mats.resized = new cv.Mat();
      mats.faces = new cv.RectVector();
      mats.size0 = new cv.Size(0, 0);
      mats.size64 = new cv.Size(64, 64);

      const totalPixels = 64 * 64;
      floatInputRef.current = new Float32Array(3 * totalPixels);

      const session = sessionRef.current;
      const labels = labelsRef.current;
      if (!session || !labels) {
        throw new Error("โมเดลยังไม่พร้อม - กรุณารีเฟรชหน้าเว็บ");
      }

      console.log("เริ่มการตรวจจับอารมณ์:", {
        sessionReady: !!session,
        labelsCount: labels.length,
        labels,
      });

      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      const inferEveryMs = 250; // ~4 FPS

      const loop = async (ts: number) => {
        rafRef.current = requestAnimationFrame((t) => {
          // eslint-disable-next-line @typescript-eslint/no-floating-promises
          loop(t);
        });

        if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) return;

        // draw current video frame on canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        mats.src.data.set(imageData.data);

        cv.cvtColor(mats.src, mats.gray, cv.COLOR_RGBA2GRAY);

        mats.faces.delete?.();
        mats.faces = new cv.RectVector();
        cascade.detectMultiScale(mats.gray, mats.faces, 1.1, 3, 0, mats.size0, mats.size0);

        if (mats.faces.size() > 0) {
          const face = mats.faces.get(0);
          const rectColor = new cv.Scalar(0, 255, 0, 255);
          cv.rectangle(
            mats.src,
            new cv.Point(face.x, face.y),
            new cv.Point(face.x + face.width, face.y + face.height),
            rectColor,
            2,
          );

          cv.cvtColor(mats.src, mats.rgb, cv.COLOR_RGBA2RGB);
          const roi = mats.rgb.roi(face);
          roi.copyTo(mats.face);
          roi.delete();

          // Resize to 64x64 for model (โมเดลใหม่ต้องการ 64x64 ไม่ใช่ 224x224)
          cv.resize(mats.face, mats.resized, mats.size64, 0, 0, cv.INTER_AREA);

          // Run inference at a throttled interval and never overlap runs
          if (!isInferBusyRef.current && ts - lastInferTsRef.current >= inferEveryMs) {
            isInferBusyRef.current = true;
            lastInferTsRef.current = ts;

            const floatData = floatInputRef.current!;
            // mats.resized is RGB (3 channels): data layout is [R,G,B, R,G,B, ...]
            // We need to convert to CHW format: [1, 3, 64, 64] where channels are separated
            const resizedData = mats.resized.data;
            const channels = mats.resized.channels();
            
            if (channels !== 3) {
              console.error(`Expected 3 channels but got ${channels}`);
              isInferBusyRef.current = false;
              return;
            }
            
            // Convert from HWC (height, width, channel) to CHW (channel, height, width)
            for (let i = 0; i < totalPixels; i++) {
              const base = i * 3;
              const r = resizedData[base] / 255.0;
              const g = resizedData[base + 1] / 255.0;
              const b = resizedData[base + 2] / 255.0;
              // CHW format: [R channel (all pixels), G channel (all pixels), B channel (all pixels)]
              floatData[i] = r;
              floatData[i + totalPixels] = g;
              floatData[i + 2 * totalPixels] = b;
            }

            const inputTensor = new ort.Tensor("float32", floatData, [1, 3, 64, 64]);
            const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };

            try {
              const results = await session.run(feeds);
              const output = results[outputName] as ort.Tensor;
              const scores = output.data as Float32Array;
              
              if (!scores || scores.length === 0) {
                console.warn("โมเดลไม่คืนค่าผลลัพธ์");
                isInferBusyRef.current = false;
                return;
              }
              
              const { bestIdx, bestP } = softmaxTop1(scores);
              setEmotion({
                label: labels[bestIdx] ?? "unknown",
                confidence: bestP * 100,
              });
            } catch (inferErr) {
              console.error("Error during inference:", inferErr);
              setErrorMessage(
                inferErr instanceof Error
                  ? `การทำนายล้มเหลว: ${inferErr.message}`
                  : "การทำนายล้มเหลว",
              );
            } finally {
              isInferBusyRef.current = false;
            }
          }

          // draw updated frame with rectangle
          const outImageData = new ImageData(
            new Uint8ClampedArray(mats.src.data),
            canvas.width,
            canvas.height,
          );
          ctx.putImageData(outImageData, 0, 0);
        } else {
          setEmotion(null);
        }
      };

      // Start loop
      rafRef.current = requestAnimationFrame((t) => {
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        loop(t);
      });
    } catch (err: unknown) {
      console.error(err);
      setErrorMessage(
        err instanceof Error ? err.message : String(err),
      );
      setStatus("เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง");
      stopCamera();
    } finally {
      setIsLoading(false);
    }
  }, [loadCascade, loadModel, loadOpenCV, softmaxTop1, stopCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100 px-4 py-8 font-sans">
      {/* Big center title */}
      <div className="mb-12 text-center">
        <h1 className="text-5xl font-bold leading-tight tracking-wide text-amber-900 sm:text-6xl drop-shadow-sm">
          Build
          <br />
          <span className="text-amber-700">Web App</span>
        </h1>
      </div>

      {/* Content area */}
      <main className="flex w-full max-w-5xl justify-center">
        <section className="w-full max-w-2xl rounded-2xl border-2 border-amber-800/20 bg-gradient-to-br from-amber-100 via-amber-50 to-orange-50 px-8 pb-8 pt-6 shadow-xl backdrop-blur-sm">
          {/* Header with icon */}
          <div className="mb-6 flex items-center gap-3 border-b-2 border-amber-800/30 pb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-700 text-white shadow-md">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-bold text-amber-900">
              Face Emotion Detection
            </h2>
            <span className="ml-auto rounded-full bg-amber-700/20 px-3 py-1 text-xs font-medium text-amber-800">
              OpenCV + YOLO11-CLS
            </span>
          </div>

          {/* Status and Emotion Display */}
          <div className="mb-4 rounded-lg bg-white/60 p-4 shadow-inner backdrop-blur-sm">
            <div className="mb-3 space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-amber-800">สถานะ:</span>
                <span className="text-amber-900">{status}</span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold text-amber-800">อารมณ์:</span>
                <span className="rounded-md bg-amber-200 px-3 py-1 font-bold text-amber-900 shadow-sm">
                  {emotion
                    ? `${emotionTranslations[emotion.label] ?? emotion.label} (${emotion.label})`
                    : "-"}
                </span>
                <span className="text-amber-700">|</span>
                <span className="font-semibold text-amber-800">
                  ความมั่นใจ:
                </span>
                <span className="rounded-md bg-amber-300 px-3 py-1 font-bold text-amber-900 shadow-sm">
                  {emotion ? emotion.confidence.toFixed(1) : "0.0"}%
                </span>
              </div>
            </div>
            {errorMessage && (
              <div className="mt-2 rounded-md bg-red-100 border border-red-300 p-2">
                <p className="text-xs font-medium text-red-700">
                  {errorMessage}
                </p>
              </div>
            )}
          </div>

          {/* Control Button */}
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center">
            <button
              type="button"
              disabled={isLoading}
              onClick={isRunning ? stopCamera : startCamera}
              className={`rounded-lg px-6 py-3 text-sm font-semibold text-white shadow-lg transition-all duration-200 ${
                isRunning
                  ? "bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 hover:shadow-xl"
                  : "bg-gradient-to-r from-amber-700 to-amber-800 hover:from-amber-800 hover:to-amber-900 hover:shadow-xl"
              } disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:shadow-lg`}
            >
              {isLoading
                ? "กำลังเริ่ม..."
                : isRunning
                  ? "⏹ Stop Camera"
                  : "▶ Start Camera"}
            </button>

            <div className="flex items-center gap-2 rounded-lg bg-amber-200/50 px-3 py-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 text-amber-700"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
              <span className="text-xs font-medium text-amber-800">
                กล้องจะทำงานบนเบราว์เซอร์เท่านั้น และไม่ส่งข้อมูลไปยังเซิร์ฟเวอร์
              </span>
            </div>
          </div>

          {/* Video & canvas */}
          <div className="mb-3 overflow-hidden rounded-xl border-4 border-amber-800/30 bg-gradient-to-br from-amber-200 to-orange-200 shadow-inner">
            <video
              ref={videoRef}
              className="hidden"
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="block h-64 w-full object-cover"
            />
          </div>

          {/* Footer Note */}
          <div className="rounded-lg bg-amber-200/40 px-3 py-2">
            <p className="text-xs font-medium text-amber-800">
              💡 หมายเหตุ: ต้องกดปุ่ม Start เพื่อขอสิทธิ์เปิดกล้อง
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
