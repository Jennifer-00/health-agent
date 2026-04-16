"use client";

interface Props {
  text: string;
  onDismiss: () => void;
}

export default function EmergencyModal({ text, onDismiss }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* 半透明遮罩 */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onDismiss} />

      {/* 弹窗主体 */}
      <div className="relative w-full max-w-md rounded-2xl bg-white shadow-2xl overflow-hidden">
        {/* 红色顶栏 */}
        <div className="bg-red-600 px-6 py-4 flex items-center gap-3">
          <span className="text-2xl">🚨</span>
          <div>
            <p className="text-white font-bold text-base leading-tight">紧急医疗提示</p>
            <p className="text-red-200 text-xs mt-0.5">请立即采取行动</p>
          </div>
          {/* 闪烁圆点 */}
          <span className="ml-auto h-3 w-3 rounded-full bg-red-300 animate-ping" />
        </div>

        {/* 正文 */}
        <div className="px-6 py-5">
          <p className="text-slate-700 text-sm leading-7 whitespace-pre-wrap">{text}</p>
        </div>

        {/* 操作按钮 */}
        <div className="px-6 pb-5 flex flex-col gap-2">
          <a
            href="tel:120"
            className="flex items-center justify-center gap-2 w-full rounded-xl bg-red-600 py-3 text-white font-bold text-sm hover:bg-red-700 transition"
          >
            <span>📞</span> 立即拨打 120
          </a>
          <button
            onClick={onDismiss}
            className="w-full rounded-xl border border-slate-200 py-2.5 text-slate-500 text-sm hover:bg-slate-50 transition"
          >
            我已知晓，继续对话
          </button>
        </div>
      </div>
    </div>
  );
}
