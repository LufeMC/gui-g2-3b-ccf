export interface Example {
  name: string;
  instruction: string;
  dot: { x: number; y: number };
  confidence: number;
  emoji: string;
  color: string;
  svg: string;
}

function buildMockSVG(type: string): string {
  const w = 800;
  const h = 520;
  let content = "";

  if (type === "google") {
    content = `<rect width="${w}" height="${h}" fill="#fff"/>
      <text x="400" y="190" text-anchor="middle" font-family="Arial,sans-serif" font-size="68" font-weight="bold"><tspan fill="#4285F4">G</tspan><tspan fill="#EA4335">o</tspan><tspan fill="#FBBC05">o</tspan><tspan fill="#4285F4">g</tspan><tspan fill="#34A853">l</tspan><tspan fill="#EA4335">e</tspan></text>
      <rect x="160" y="240" width="480" height="48" rx="24" fill="#fff" stroke="#ddd" stroke-width="1.5"/>
      <text x="400" y="270" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#999">Search Google or type a URL</text>
      <circle cx="190" cy="264" r="10" fill="none" stroke="#4285F4" stroke-width="2"/><line x1="197" y1="271" x2="204" y2="278" stroke="#4285F4" stroke-width="2"/>
      <rect x="300" y="330" width="90" height="36" rx="4" fill="#f3f4f6"/><text x="345" y="353" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#555">Google Search</text>
      <rect x="410" y="330" width="100" height="36" rx="4" fill="#f3f4f6"/><text x="460" y="353" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#555">I'm Feeling Lucky</text>`;
  } else if (type === "mobile") {
    content = `<rect width="${w}" height="${h}" fill="#f9fafb"/>
      <rect x="0" y="0" width="${w}" height="56" fill="#fff"/>
      <text x="24" y="36" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#111">My App</text>
      <circle cx="${w - 36}" cy="28" r="14" fill="#f3f4f6" stroke="#ddd" stroke-width="1"/><text x="${w - 36}" y="33" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#555">\u2699</text>
      <rect x="24" y="80" width="${w - 48}" height="66" rx="12" fill="#fff" stroke="#e5e7eb"/><text x="44" y="108" font-family="Arial,sans-serif" font-size="15" font-weight="600" fill="#111">Profile</text><text x="44" y="128" font-family="Arial,sans-serif" font-size="12" fill="#888">Manage your account settings</text>
      <rect x="24" y="160" width="${w - 48}" height="66" rx="12" fill="#fff" stroke="#e5e7eb"/><text x="44" y="188" font-family="Arial,sans-serif" font-size="15" font-weight="600" fill="#111">Notifications</text><text x="44" y="208" font-family="Arial,sans-serif" font-size="12" fill="#888">Configure alert preferences</text>
      <rect x="24" y="240" width="${w - 48}" height="66" rx="12" fill="#fff" stroke="#e5e7eb"/><text x="44" y="268" font-family="Arial,sans-serif" font-size="15" font-weight="600" fill="#111">Privacy</text><text x="44" y="288" font-family="Arial,sans-serif" font-size="12" fill="#888">Data sharing and permissions</text>`;
  } else if (type === "dashboard") {
    const bars = Array.from({ length: 12 }, (_, i) => {
      const bx = 80 + i * 60;
      const bh = 40 + Math.sin(i * 0.8 + 1) * 60 + Math.random() * 30;
      return `<rect x="${bx}" y="${380 - bh}" width="36" height="${bh}" rx="4" fill="#3b82f6" opacity="0.8"/>`;
    }).join("");
    content = `<rect width="${w}" height="${h}" fill="#f9fafb"/>
      <rect x="0" y="0" width="${w}" height="56" fill="#1e293b"/>
      <text x="24" y="36" font-family="Arial,sans-serif" font-size="16" font-weight="bold" fill="#fff">Analytics Dashboard</text>
      <rect x="${w - 120}" y="14" width="96" height="30" rx="6" fill="#3b82f6"/><text x="${w - 72}" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="600" fill="#fff">\u2B07 Export</text>
      <rect x="24" y="80" width="240" height="100" rx="12" fill="#fff" stroke="#e5e7eb"/><text x="44" y="108" font-family="Arial,sans-serif" font-size="12" fill="#888">Total Users</text><text x="44" y="140" font-family="Arial,sans-serif" font-size="28" font-weight="bold" fill="#111">24,521</text><text x="44" y="164" font-family="Arial,sans-serif" font-size="12" fill="#22c55e">\u25B2 12.3%</text>
      <rect x="280" y="80" width="240" height="100" rx="12" fill="#fff" stroke="#e5e7eb"/><text x="300" y="108" font-family="Arial,sans-serif" font-size="12" fill="#888">Revenue</text><text x="300" y="140" font-family="Arial,sans-serif" font-size="28" font-weight="bold" fill="#111">$84.2K</text><text x="300" y="164" font-family="Arial,sans-serif" font-size="12" fill="#22c55e">\u25B2 8.1%</text>
      <rect x="536" y="80" width="240" height="100" rx="12" fill="#fff" stroke="#e5e7eb"/><text x="556" y="108" font-family="Arial,sans-serif" font-size="12" fill="#888">Conversion</text><text x="556" y="140" font-family="Arial,sans-serif" font-size="28" font-weight="bold" fill="#111">3.42%</text><text x="556" y="164" font-family="Arial,sans-serif" font-size="12" fill="#ef4444">\u25BC 0.8%</text>
      <rect x="24" y="204" width="${w - 48}" height="260" rx="12" fill="#fff" stroke="#e5e7eb"/><line x1="60" y1="380" x2="${w - 60}" y2="380" stroke="#e5e7eb"/>${bars}`;
  } else {
    content = `<rect width="${w}" height="${h}" fill="#fff"/>
      <text x="400" y="60" text-anchor="middle" font-family="Arial,sans-serif" font-size="24" font-weight="bold" fill="#111">Create Account</text>
      <text x="400" y="88" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#888">Fill in your details to get started</text>
      <text x="220" y="140" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#333">Full Name</text><rect x="220" y="150" width="360" height="42" rx="8" fill="#fff" stroke="#d1d5db" stroke-width="1.5"/>
      <text x="220" y="220" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#333">Email</text><rect x="220" y="230" width="360" height="42" rx="8" fill="#fff" stroke="#d1d5db" stroke-width="1.5"/>
      <text x="220" y="300" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#333">Password</text><rect x="220" y="310" width="360" height="42" rx="8" fill="#fff" stroke="#d1d5db" stroke-width="1.5"/>
      <rect x="220" y="390" width="360" height="46" rx="8" fill="#E63B19"/><text x="400" y="419" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" font-weight="bold" fill="#fff">Submit</text>`;
  }

  return `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">${content}</svg>`)}`;
}

export const EXAMPLES: Example[] = [
  {
    name: "Google Homepage",
    instruction: "click the search bar",
    dot: { x: 0.5, y: 0.52 },
    confidence: 0.96,
    emoji: "\uD83D\uDD0D",
    color: "#4285F4",
    svg: buildMockSVG("google"),
  },
  {
    name: "Mobile Settings",
    instruction: "click the settings icon",
    dot: { x: 0.91, y: 0.06 },
    confidence: 0.91,
    emoji: "\uD83D\uDCF1",
    color: "#34A853",
    svg: buildMockSVG("mobile"),
  },
  {
    name: "Analytics Dashboard",
    instruction: "click the export button",
    dot: { x: 0.88, y: 0.12 },
    confidence: 0.73,
    emoji: "\uD83D\uDCCA",
    color: "#7C3AED",
    svg: buildMockSVG("dashboard"),
  },
  {
    name: "Signup Form",
    instruction: "click the submit button",
    dot: { x: 0.5, y: 0.82 },
    confidence: 0.94,
    emoji: "\uD83D\uDCDD",
    color: "#E63B19",
    svg: buildMockSVG("form"),
  },
];
