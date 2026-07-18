export default function GradientMesh({ className }: { className?: string }) {
  return (
    <div
      className={`absolute inset-0 pointer-events-none overflow-hidden ${className ?? ''}`}
      aria-hidden="true"
    >
      {/* Blue orb top left */}
      <div className="absolute -top-40 -left-40 w-96 h-96 bg-primary-from/20 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '4s' }} />
      {/* Cyan orb top right */}
      <div className="absolute -top-20 -right-20 w-80 h-80 bg-primary-to/15 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '6s', animationDelay: '1s' }} />
      {/* Purple orb bottom */}
      <div className="absolute bottom-0 left-1/3 w-72 h-72 bg-violet-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '5s', animationDelay: '2s' }} />
    </div>
  )
}
