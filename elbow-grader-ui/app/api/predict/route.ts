import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

/**
 * POST /api/predict
 * Forwards the multipart form (ap_file, lat_file, config fields) to FastAPI.
 * Returns the FastAPI JSON response or a structured error.
 */
export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
      // Let fetch set the Content-Type with the correct boundary.
    });

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: `Backend error ${res.status}`, detail: text },
        { status: res.status },
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Failed to reach Python backend", detail: message },
      { status: 502 },
    );
  }
}

/**
 * GET /api/predict → health-check proxy
 */
export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: "Backend unreachable", detail: message },
      { status: 502 },
    );
  }
}
