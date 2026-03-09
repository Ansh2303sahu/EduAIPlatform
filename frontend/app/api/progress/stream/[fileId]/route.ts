import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(req: Request, ctx: { params: { fileId: string } }) {
  const fileId = ctx.params.fileId;
  const backend = process.env.NEXT_PUBLIC_BACKEND_URL!;
  const auth = req.headers.get("authorization");

  if (!auth) return new NextResponse("Missing auth", { status: 401 });

  const r = await fetch(`${backend}/api/progress/stream/${fileId}`, {
    headers: { Authorization: auth },
  });

  return new NextResponse(r.body, {
    status: r.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}