/** PV-IQA Palm Vein Logo — scalable palm-vein biometric icon. */
export function Logo({ size = 24, className }: { size?: number; className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      className={className}
      fill="none"
      stroke="currentColor"
      // Coerce size to an approximate visual match: 24→24, 28→28, 48→48, etc.
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      {/* Outer ring — palm boundary */}
      <circle cx="24" cy="24" r="20" strokeWidth="1.2" opacity="0.35" />
      {/* Scanning circle — NIR illumination area */}
      <circle cx="24" cy="24" r="14" strokeWidth="1" opacity="0.55" />
      {/* Vein pattern arcs */}
      <path
        d="M17 15 Q23 20 19 26 Q16 30 22 34"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <path
        d="M27 14 Q31 18 29 24 Q27 30 33 34"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <path
        d="M13 22 Q17 24 19 28"
        strokeWidth="0.9"
        strokeLinecap="round"
        opacity="0.7"
      />
      <path
        d="M35 20 Q31 24 29 28"
        strokeWidth="0.9"
        strokeLinecap="round"
        opacity="0.7"
      />
      {/* Center scan dot */}
      <circle cx="24" cy="24" r="2.5" fill="currentColor" stroke="none" opacity="0.85" />
      {/* Dashed scan ring */}
      <circle
        cx="24"
        cy="24"
        r="5"
        strokeWidth="0.6"
        opacity="0.3"
        strokeDasharray="3 2"
      />
      {/* Scan crosshairs */}
      <path d="M24 5v4" strokeWidth="0.6" opacity="0.25" />
      <path d="M24 39v4" strokeWidth="0.6" opacity="0.25" />
      <path d="M5 24h4" strokeWidth="0.6" opacity="0.25" />
      <path d="M39 24h4" strokeWidth="0.6" opacity="0.25" />
    </svg>
  )
}
