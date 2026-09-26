export function BrandPanel() {
  return (
    <section dir="rtl" className="relative h-full min-h-[610px] overflow-hidden text-white">
      {/* Existing geometric layers */}
      <div aria-hidden="true" className="pointer-events-none absolute right-[8%] top-[-5%] h-[47%] w-[66%] rounded-[0_48px_70px_70px] bg-white/[0.035]" />
      <div aria-hidden="true" className="pointer-events-none absolute right-[-13%] top-[-1%] h-[61%] w-[70%] rounded-bl-[95px] bg-black/[0.08]" />

      {/* Oversized ornament blended into the entire right edge */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -right-[31%] -top-[9%] h-[122%] w-[92%] bg-contain bg-right bg-no-repeat opacity-[0.075]"
        style={{
          backgroundImage: "url('/images/arabic-ornament.png')",
          WebkitMaskImage: "linear-gradient(to left, #000 0%, #000 40%, rgba(0,0,0,.72) 63%, transparent 100%)",
          maskImage: "linear-gradient(to left, #000 0%, #000 40%, rgba(0,0,0,.72) 63%, transparent 100%)",
        }}
      />

      {/* Logo moved toward the upper-right corner */}
      <div className="pointer-events-none absolute right-[0.5%] top-[1.5%] z-10 flex h-[31%] w-[62%] items-start justify-end pt-5 pr-5">
        <img src="/images/ramzah-logo.png" alt="رمزة" className="w-[72%] max-w-[315px] object-contain" />
      </div>

      <div className="absolute bottom-[5%] right-[5%] left-[5%] z-10 text-right leading-relaxed">
        <p className="text-[16px] font-medium sm:text-[17px]">بإشارتك، يبدأ التواصل</p>
        <p className="mt-1 max-w-[470px] text-[13px] font-normal leading-7 text-white/90 sm:text-[14px]">
          تحوّل رَمزَة لغة الإشارة السعودية إلى نص عربي وصوت مسموع، لتعبّر بطريقتك وتتواصل بسهولة واستقلالية.
        </p>
      </div>
    </section>
  )
}
