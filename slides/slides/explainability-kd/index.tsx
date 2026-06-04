import type { DesignSystem, Page, SlideMeta, SlideTransition } from '@open-slide/core';
import { useSlidePageNumber } from '@open-slide/core';

// === Figures ===
import pipelineImg from './assets/pipeline.png';
import probesImg from './assets/probes.png';
import qualBoleteImg from './assets/qual_case_bolete.png';
import qualMixingBowlImg from './assets/qual_case_mixing_bowl.png';
import qualTriceratopsSceneImg from './assets/qual_case_triceratops_scene.png';
import qualTriceratopsSkeletonImg from './assets/qual_case_triceratops_skeleton.png';
import qualitativeCaseText from './assets/qualitative_case_text.json';

// === Design system ===
export const design: DesignSystem = {
  palette: {
    bg: '#fdf5ec',
    text: '#2c1810',
    accent: '#d4602a',
  },
  fonts: {
    display: 'Georgia, "Times New Roman", serif',
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
  },
  typeScale: {
    hero: 64,
    body: 36,
  },
  radius: 8,
};

const amber = '#e8923b';
const crimson = '#c0392b';
const muted = '#8c7b6c';

const fill = {
  width: '100%',
  height: '100%',
  fontFamily: 'var(--osd-font-body)',
} as const;

// === Keyframes (injected once by FlameBar, which appears on every page) ===
const keyframesCSS = `
@keyframes riseIn { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideUp { from { opacity: 0; transform: translateY(36px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes pulseGlow { 0%,100% { opacity: 0.35; } 50% { opacity: 0.7; } }
`;

// === Animation helpers ===
const riseIn = (delay: number): React.CSSProperties =>
  ({ animation: `riseIn 0.7s ease-out ${delay}s both` });

const fadeIn = (delay: number): React.CSSProperties =>
  ({ animation: `fadeIn 1s ease-out ${delay}s both` });

// === Page number footer ===
const Footer = () => {
  const { current, total } = useSlidePageNumber();
  return (
    <div style={{ position: 'absolute', bottom: 40, right: 120, fontSize: 22, color: muted }}>
      {String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}
    </div>
  );
};

// === Decorative flame accent bar (carries keyframes for the whole deck) ===
const FlameBar = () => (
  <>
    <style>{keyframesCSS}</style>
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 5,
        background: `linear-gradient(90deg, ${amber}, ${crimson}, ${amber})`,
        backgroundSize: '200% 100%',
        animation: 'gradientShift 5s ease-in-out infinite',
      }}
    />
  </>
);

// === Shared eyebrow ===
const Eyebrow = ({ children, delay = 0.1 }: { children: string; delay?: number }) => (
  <div style={{ fontSize: 24, color: 'var(--osd-accent)', letterSpacing: '0.22em', textTransform: 'uppercase', marginBottom: 28, ...riseIn(delay) }}>
    {children}
  </div>
);

// === Shared page heading ===
const PageHeading = ({ children, delay = 0.15 }: { children: string; delay?: number }) => (
  <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 80, fontWeight: 800, margin: 0, lineHeight: 1.15, ...riseIn(delay) }}>
    {children}
  </h2>
);

// === Bullet with amber dot ===
const Bullet = ({ children, delay = 0 }: { children: string | React.ReactNode; delay?: number }) => (
  <li style={{ fontSize: 'var(--osd-size-body)', lineHeight: 1.55, marginBottom: 28, listStyle: 'none', position: 'relative', ...riseIn(delay) }}>
    {children}
  </li>
);

// === Easing ===
const EASE_OUT = 'cubic-bezier(0, 0, 0.2, 1)';
const EASE_IN = 'cubic-bezier(0.4, 0, 1, 1)';

// ================================================================
// PAGE 1 — Cover
// ================================================================
const Cover: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '0 160px', position: 'relative', overflow: 'hidden' }}>
    <FlameBar />
    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: 900, height: 500, borderRadius: '50%', background: `radial-gradient(ellipse, ${amber}15, transparent 70%)`, animation: 'pulseGlow 4s ease-in-out infinite', pointerEvents: 'none' }} />
    <div style={{ textAlign: 'center', maxWidth: 1300, position: 'relative', zIndex: 1 }}>
      <h1 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 'var(--osd-size-hero)', fontWeight: 900, margin: 0, lineHeight: 1.12, ...riseIn(0.25) }}>
        Look Where It Matters:<br />Distilling Vision Through Explanations
      </h1>
      <p style={{ fontSize: 34, color: muted, marginTop: 44, lineHeight: 1.5, ...riseIn(0.5) }}>
        Can explainability provide a useful training signal<br />for distilling vision-language models?
      </p>
      <div style={{ marginTop: 56, ...riseIn(0.75) }}>
        <span style={{ fontSize: 28, color: 'var(--osd-accent)', fontWeight: 600 }}>Mattia Curri</span>
      </div>
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 2 — Motivation
// ================================================================
const Motivation: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Motivation</Eyebrow>
    <PageHeading>Can explanations become a<br />training signal?</PageHeading>
    <p style={{ fontSize: 38, color: muted, lineHeight: 1.5, maxWidth: 1300, marginTop: 48, marginBottom: 0, ...riseIn(0.3) }}>
      Knowledge distillation usually treats the teacher signal as a target to be matched <strong style={{ color: 'var(--osd-text)' }}>uniformly</strong>{''}
    </p>
    <ul style={{ marginTop: 52, marginBottom: 0, padding: 0, maxWidth: 1300 }}>
      <Bullet delay={0.45}>What if we <strong style={{ color: 'var(--osd-accent)' }}>weight</strong> the teacher signal by estimated importance?</Bullet>
      <Bullet delay={0.6}>Grad-CAM saliency tells us <strong style={{ color: 'var(--osd-accent)' }}>which visual tokens</strong> the teacher relied on most.</Bullet>
    </ul>
    <Footer />
  </div>
);

// ================================================================
// PAGE 3 — KD vs Our approach (uniform cards)
// ================================================================
const Approach: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>The Approach</Eyebrow>
    <PageHeading>From uniform matching to<br />saliency-weighted distillation</PageHeading>
    <div style={{ display: 'flex', gap: 48, marginTop: 60, maxWidth: 1480, alignItems: 'stretch' }}>
      {/* Standard KD */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', ...riseIn(0.35) }}>
        <div style={{ fontSize: 26, color: muted, letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 24 }}>Standard KD</div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', background: 'rgba(140,123,108,0.08)', borderRadius: 'var(--osd-radius)', padding: '52px 44px', borderLeft: `4px solid ${muted}` }}>
          <p style={{ fontSize: 35, lineHeight: 1.5, margin: 0, color: muted }}>Match every teacher feature equally, with no notion of which features actually drive the teacher's decisions.</p>
        </div>
      </div>
      {/* Divider */}
      <div style={{ display: 'flex', alignItems: 'center', flexShrink: 0, ...fadeIn(0.55) }}>
        <div style={{ width: 44, height: 44, borderRadius: '50%', border: `3px solid ${amber}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 28, color: amber, fontWeight: 700 }}>→</div>
      </div>
      {/* Our approach */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', ...riseIn(0.65) }}>
        <div style={{ fontSize: 26, color: 'var(--osd-accent)', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 24 }}>Our approach</div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', background: `linear-gradient(135deg, ${amber}0F, ${crimson}0F)`, borderRadius: 'var(--osd-radius)', padding: '52px 44px', borderLeft: `4px solid var(--osd-accent)` }}>
          <p style={{ fontSize: 35, lineHeight: 1.5, margin: 0, color: 'var(--osd-text)' }}>
            {'Use Grad-CAM to weight tokens; the student focuses on preserving what the teacher considers '}<strong style={{ color: 'var(--osd-accent)' }}>decision-relevant</strong>.
          </p>
        </div>
      </div>
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 4 — Pipeline (image only)
// ================================================================
const Pipeline: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: '80px 120px 60px', position: 'relative', display: 'flex', flexDirection: 'column' }}>
    <FlameBar />
    <Eyebrow>Method</Eyebrow>
    <PageHeading>Pipeline overview</PageHeading>
    <div style={{ marginTop: 32, ...fadeIn(0.35), flex: 1, display: 'flex', alignItems: 'center' }}>
      <img src={pipelineImg} style={{ width: '100%', maxHeight: 720, objectFit: 'contain', borderRadius: 8 }} alt="Pipeline overview" />
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 5 — Teacher probes (image only)
// ================================================================
const TeacherProbes: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: '80px 80px 60px', position: 'relative', display: 'flex', flexDirection: 'column' }}>
    <FlameBar />
    <Eyebrow>Method</Eyebrow>
    <PageHeading>Teacher probe architectures</PageHeading>
    <div style={{ marginTop: 10, ...fadeIn(0.35), flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <img src={probesImg} style={{ width: '100%', maxHeight: 760, objectFit: 'contain', borderRadius: 8 }} alt="Probe architectures" />
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 6 — Explanation signal as a visual process
// ================================================================
const TokenMap = ({ mode }: { mode: 'tokens' | 'saliency' }) => {
  const emphasis = mode === 'saliency'
    ? [0.10, 0.16, 0.28, 0.12, 0.18, 0.82, 0.94, 0.20, 0.12, 0.62, 0.88, 0.16, 0.08, 0.14, 0.32, 0.10]
    : Array.from({ length: 16 }, (_, index) => 0.16 + (index % 4) * 0.05);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 38px)', gap: 7, marginTop: 26 }}>
      {emphasis.map((value, index) => (
        <div
          key={index}
          style={{
            width: 38,
            height: 38,
            borderRadius: 5,
            background: mode === 'tokens'
              ? `rgba(140, 123, 108, ${value})`
              : `rgba(212, 96, 42, ${value})`,
            border: mode === 'tokens' ? '1px solid rgba(140, 123, 108, 0.26)' : '1px solid rgba(212, 96, 42, 0.20)',
          }}
        />
      ))}
    </div>
  );
};

const SmallHeatmap = ({ values, columns, size, gap }: { values: number[]; columns: number; size: number; gap: number }) => (
  <div style={{ display: 'grid', gridTemplateColumns: `repeat(${columns}, ${size}px)`, gap }}>
    {values.map((value, index) => (
      <div
        key={index}
        style={{
          width: size,
          height: size,
          borderRadius: 3,
          background: `rgba(212, 96, 42, ${value})`,
          border: '1px solid rgba(212, 96, 42, 0.15)',
        }}
      />
    ))}
  </div>
);

const AlignmentMap = () => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 18, marginTop: 27 }}>
    <div>
      <div style={{ fontSize: 17, color: muted, marginBottom: 8 }}>Qwen / 28 x 28</div>
      <SmallHeatmap
        columns={6}
        size={19}
        gap={3}
        values={[0.10, 0.12, 0.16, 0.14, 0.08, 0.08, 0.12, 0.18, 0.38, 0.55, 0.24, 0.10, 0.10, 0.30, 0.78, 0.92, 0.56, 0.12, 0.08, 0.26, 0.62, 0.84, 0.42, 0.10, 0.07, 0.12, 0.21, 0.27, 0.14, 0.07, 0.06, 0.08, 0.10, 0.10, 0.08, 0.06]}
      />
    </div>
    <div style={{ color: 'var(--osd-accent)', textAlign: 'center', fontSize: 17, fontWeight: 700, lineHeight: 1.25 }}>
      resize<br />
      <span style={{ fontSize: 31, lineHeight: 1 }}>&#8594;</span>
    </div>
    <div>
      <div style={{ fontSize: 17, color: muted, marginBottom: 8 }}>ViT / 14 x 14</div>
      <SmallHeatmap
        columns={3}
        size={37}
        gap={5}
        values={[0.12, 0.36, 0.14, 0.22, 0.90, 0.34, 0.08, 0.22, 0.10]}
      />
    </div>
  </div>
);

const SignalStage = ({
  index,
  title,
  detail,
  mode,
  delay,
}: {
  index: string;
  title: string;
  detail: string;
  mode: 'tokens' | 'saliency' | 'align';
  delay: number;
}) => (
  <div style={{ flex: 1, minHeight: 350, padding: '30px 34px', borderTop: `3px solid ${mode === 'tokens' ? muted : amber}`, background: mode === 'tokens' ? 'rgba(140,123,108,0.045)' : 'rgba(212,96,42,0.045)', ...riseIn(delay) }}>
    <div style={{ color: mode === 'tokens' ? muted : 'var(--osd-accent)', fontSize: 19, fontWeight: 700, letterSpacing: '0.13em', textTransform: 'uppercase' }}>{index}</div>
    <h3 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 39, lineHeight: 1.18, margin: '17px 0 9px', fontWeight: 700 }}>{title}</h3>
    <div style={{ fontSize: 23, color: muted }}>{detail}</div>
    {mode === 'align' ? <AlignmentMap /> : <TokenMap mode={mode} />}
  </div>
);

const GradCAM: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Method</Eyebrow>
    <PageHeading>From explanation to token weights</PageHeading>
    <div style={{ display: 'flex', gap: 22, alignItems: 'stretch', marginTop: 52, maxWidth: 1510 }}>
      <SignalStage index="01 / frozen" title="Visual tokens" detail="Qwen map: 28 x 28" mode="tokens" delay={0.30} />
      <SignalStage index="02 / attribute" title="Probe saliency" detail="Class-relevant regions" mode="saliency" delay={0.43} />
      <SignalStage index="03 / align" title="Map alignment" detail="Saliency follows the new grid" mode="align" delay={0.56} />
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 7 — Objectives
// ================================================================
const LossName = ({ subscript }: { subscript?: string }) => (
  subscript
    ? <msub><mi mathvariant="script">&#x2112;</mi><mtext>{subscript}</mtext></msub>
    : <mi mathvariant="script">&#x2112;</mi>
);

const IndexedVariable = ({ value }: { value: string }) => (
  <msub><mi>{value}</mi><mi>i</mi></msub>
);

const Sum = () => (
  <munderover>
    <mo>&#x2211;</mo>
    <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow>
    <mi>N</mi>
  </munderover>
);

const NormSquared = ({ children }: { children: React.ReactNode }) => (
  <msubsup>
    <mrow><mo>&#x2225;</mo>{children}<mo>&#x2225;</mo></mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
);

const equationStyle = { fontSize: 46, color: 'var(--osd-text)', whiteSpace: 'nowrap' };

const Equation = ({ objective }: { objective: 'global' | 'explanation' | 'combined' }) => {
  if (objective === 'global') {
    return (
      <math display="inline" aria-label="Global MSE loss" style={equationStyle}>
        <mrow>
          <LossName subscript="global" /><mo>=</mo>
          <NormSquared>
            <mfrac><mn>1</mn><mi>N</mi></mfrac><Sum /><IndexedVariable value="s" />
            <mo>&#x2212;</mo>
            <mfrac><mn>1</mn><mi>N</mi></mfrac><Sum /><IndexedVariable value="t" />
          </NormSquared>
        </mrow>
      </math>
    );
  }

  if (objective === 'explanation') {
    return (
      <math display="inline" aria-label="Explanation-weighted loss" style={equationStyle}>
        <mrow>
          <LossName subscript="expl" /><mo>=</mo><Sum />
          <msub><mi>A</mi><mi>i</mi></msub>
          <NormSquared><IndexedVariable value="s" /><mo>&#x2212;</mo><IndexedVariable value="t" /></NormSquared>
        </mrow>
      </math>
    );
  }

  return (
    <math display="inline" aria-label="Combined loss" style={equationStyle}>
      <mrow>
        <LossName /><mo>=</mo>
        <msub><mi>&#x3BB;</mi><mi>g</mi></msub><LossName subscript="global" />
        <mo>+</mo>
        <msub><mi>&#x3BB;</mi><mi>e</mi></msub><LossName subscript="expl" />
      </mrow>
    </math>
  );
};

const ObjectiveRow = ({
  title,
  subtitle,
  objective,
  accent,
  delay,
}: {
  title: string;
  subtitle: string;
  objective: 'global' | 'explanation' | 'combined';
  accent: string;
  delay: number;
}) => (
  <div style={{ display: 'grid', gridTemplateColumns: '420px 1fr', alignItems: 'center', minHeight: 150, background: `linear-gradient(90deg, ${accent}0D, ${accent}03)`, borderRadius: 'var(--osd-radius)', padding: '24px 36px 24px 32px', borderLeft: `4px solid ${accent}`, ...riseIn(delay) }}>
    <div>
      <div style={{ fontSize: 20, color: muted, letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 9 }}>{subtitle}</div>
      <h3 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 26, fontWeight: 800, margin: 0, lineHeight: 1.18, whiteSpace: 'nowrap' }}>{title}</h3>
    </div>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minWidth: 0, background: 'transparent' }}>
      <Equation objective={objective} />
    </div>
  </div>
);

const Objectives: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Method</Eyebrow>
    <PageHeading>Three distillation objectives</PageHeading>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18, marginTop: 42, maxWidth: 1500 }}>
      <ObjectiveRow title="Global MSE" subtitle="Baseline" objective="global" accent={muted} delay={0.3} />
      <ObjectiveRow title="Explanation-weighted" subtitle="Saliency-guided" objective="explanation" accent={amber} delay={0.45} />
      <ObjectiveRow title="Combined" subtitle="Global + local" objective="combined" accent={crimson} delay={0.6} />
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 8 — Probe results (table, not unused image)
// ================================================================
const ProbeResults: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Results</Eyebrow>
    <PageHeading>Teacher probe accuracy</PageHeading>
    {/* Table */}
    <div style={{ margin: '64px auto 0', width: 1200, maxWidth: '100%', ...riseIn(0.35) }}>
      <div style={{ display: 'flex', borderBottom: `2px solid ${muted}40`, paddingBottom: 20, marginBottom: 20 }}>
        <div style={{ flex: 1.4, fontSize: 26, color: muted, fontWeight: 600 }}>Probe type</div>
        <div style={{ flex: 0.8, fontSize: 26, color: muted, fontWeight: 600, textAlign: 'center' }}>Best epoch</div>
        <div style={{ flex: 1, fontSize: 26, color: muted, fontWeight: 600, textAlign: 'center' }}>Val accuracy</div>
        <div style={{ flex: 1, fontSize: 26, color: muted, fontWeight: 600, textAlign: 'center' }}>Test accuracy</div>
      </div>
      <Row label="Mean" epoch="10" val="95.07" test="95.00" bestVal bestTest delay={0.45} />
      <Row label="Attention" epoch="2" val="94.93" test="94.48" delay={0.55} />
      <Row label="Cross-attn" epoch="10" val="94.58" test="94.78" delay={0.65} />
    </div>
    <Footer />
  </div>
);

const Row = ({ label, epoch, val, test, bestVal, bestTest, delay }: { label: string; epoch: string; val: string; test: string; bestVal?: boolean; bestTest?: boolean; delay: number }) => (
  <div style={{ display: 'flex', padding: '28px 0', borderBottom: `1px solid ${muted}15`, ...riseIn(delay) }}>
    <div style={{ flex: 1.4, fontSize: 34, fontFamily: 'var(--osd-font-display)', fontWeight: 400, color: '#2c1810' }}>
      {label}
    </div>
    <div style={{ flex: 0.8, fontSize: 34, textAlign: 'center', color: 'var(--osd-text)', fontWeight: 400 }}>
      {epoch}
    </div>
    <div style={{ flex: 1, fontSize: 34, textAlign: 'center', color: 'var(--osd-text)', fontWeight: bestVal ? 800 : 400, textDecoration: bestVal ? 'underline' : 'none', textUnderlineOffset: 8 }}>
      {val}%
    </div>
    <div style={{ flex: 1, fontSize: 34, textAlign: 'center', color: 'var(--osd-text)', fontWeight: bestTest ? 800 : 400, textDecoration: bestTest ? 'underline' : 'none', textUnderlineOffset: 8 }}>
      {test}%
    </div>
  </div>
);

// ================================================================
// PAGE 9 — Primary results (table, not unused image)
// ================================================================
const PrimaryResults: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Results</Eyebrow>
    <PageHeading>LLM jury{''}</PageHeading>
    <div style={{ marginTop: 52, ...riseIn(0.35) }}>
      <div style={{ display: 'flex', borderBottom: `2px solid ${muted}40`, paddingBottom: 16, marginBottom: 16 }}>
        <div style={{ flex: 1.6, fontSize: 23, color: muted, fontWeight: 600 }}>Variant</div>
        <div style={{ flex: 0.8, fontSize: 23, color: muted, fontWeight: 600, textAlign: 'center' }}>Student</div>
        <div style={{ flex: 0.8, fontSize: 23, color: muted, fontWeight: 600, textAlign: 'center' }}>Teacher</div>
        <div style={{ flex: 0.8, fontSize: 23, color: muted, fontWeight: 600, textAlign: 'center' }}>All agree</div>
        <div style={{ flex: 0.6, fontSize: 23, color: muted, fontWeight: 600, textAlign: 'center' }}>κ</div>
      </div>
      <ResRow label="Base MSE" student="10" teacher="427" agree="0.936" k="0.312" bestAgree delay={0.42} />
      <ResRow label="Mean, expl-only" student="69" teacher="367" agree="0.705" k="0.355" bestStudent bestTeacher delay={0.5} />
      <ResRow label="Attn, expl-only" student="56" teacher="381" agree="0.741" k="0.367" delay={0.58} />
      <ResRow label="Cross, expl-only" student="20" teacher="416" agree="0.856" k="0.337" delay={0.66} />
      <ResRow label="Mean, MSE+expl" student="58" teacher="379" agree="0.737" k="0.355" delay={0.74} />
      <ResRow label="Attn, MSE+expl" student="61" teacher="376" agree="0.737" k="0.338" delay={0.82} />
      <ResRow label="Cross, MSE+expl" student="37" teacher="400" agree="0.817" k="0.372" bestK delay={0.9} />
    </div>
    <Footer />
  </div>
);

const ResRow = ({ label, student, teacher, agree, k, bestLabel, bestStudent, bestTeacher, bestAgree, bestK, delay }: { label: string; student: string; teacher: string; agree: string; k: string; bestLabel?: boolean; bestStudent?: boolean; bestTeacher?: boolean; bestAgree?: boolean; bestK?: boolean; delay: number }) => {
  const best = { fontWeight: 800, textDecoration: 'underline', textUnderlineOffset: 6 } as const;
  return (
  <div style={{ display: 'flex', padding: '18px 0', borderBottom: `1px solid ${muted}10`, ...riseIn(delay) }}>
    <div style={{ flex: 1.6, fontSize: 26, fontFamily: 'var(--osd-font-display)', fontWeight: bestLabel ? 800 : 500, color: bestLabel ? 'var(--osd-accent)' : 'var(--osd-text)' }}>{label}</div>
    <div style={{ flex: 0.8, fontSize: 26, textAlign: 'center', color: 'var(--osd-text)', ...(bestStudent ? best : {}) }}>{student}</div>
    <div style={{ flex: 0.8, fontSize: 26, textAlign: 'center', color: muted, ...(bestTeacher ? best : {}) }}>{teacher}</div>
    <div style={{ flex: 0.8, fontSize: 26, textAlign: 'center', color: 'var(--osd-text)', ...(bestAgree ? best : {}) }}>{agree}</div>
    <div style={{ flex: 0.6, fontSize: 26, textAlign: 'center', color: 'var(--osd-text)', ...(bestK ? best : {}) }}>{k}</div>
  </div>
)};

// ================================================================
// PAGE 10 — Judge tendencies (as text summary, not unused image)
// ================================================================
const JudgeTendencies: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Results</Eyebrow>
    <PageHeading>Every judge shifts toward<br />the explanation-guided student</PageHeading>
    <div style={{ marginTop: 64, display: 'flex', gap: 48, maxWidth: 1450 }}>
      <JudgeCard name="Qwen3.5-9B" from={14} to={76} delay={0.35} />
      <JudgeCard name="GLM-4.6V-Flash" from={17} to={92} delay={0.5} />
      <JudgeCard name="MiniCPM-V-4.5" from={10} to={78} delay={0.65} />
    </div>
    <p style={{ fontSize: 28, color: muted, marginTop: 56, lineHeight: 1.5, ...riseIn(0.8) }}>
      Student-preference counts rise from baseline MSE → mean explanation-only.<br />Ties are nearly absent across all judges.
    </p>
    <Footer />
  </div>
);

const JudgeCard = ({ name, from, to, delay }: { name: string; from: number; to: number; delay: number }) => (
  <div style={{ flex: 1, background: `rgba(140,123,108,0.06)`, borderRadius: 'var(--osd-radius)', padding: '48px 40px', textAlign: 'center', ...riseIn(delay) }}>
    <div style={{ fontSize: 28, color: muted, marginBottom: 32 }}>{name}</div>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24 }}>
      <div style={{ fontSize: 22, color: muted, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Baseline</div>
      <div style={{ fontSize: 56, fontWeight: 900, color: muted }}>{from}</div>
      <div style={{ fontSize: 32, color: amber }}>→</div>
      <div style={{ fontSize: 56, fontWeight: 900, color: 'var(--osd-accent)' }}>{to}</div>
      <div style={{ fontSize: 22, color: muted, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Expl.</div>
    </div>
  </div>
);

// ================================================================
// PAGE 11 — Qualitative successes (image L + text R)
// ================================================================
type QualitativeComparison = {
  baseline: { label: string; excerpt: string };
  guided: { label: string; excerpt: string };
  reading: string;
};

const qualitativeText = qualitativeCaseText as Record<string, QualitativeComparison>;

const ResponseExcerpt = ({
  response,
  guided = false,
}: {
  response: { label: string; excerpt: string };
  guided?: boolean;
}) => (
  <div style={{ marginTop: 12 }}>
    <div style={{ fontSize: 18, fontWeight: 700, color: guided ? 'var(--osd-accent)' : muted, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
      {response.label}
    </div>
    <p style={{ margin: '5px 0 0', fontSize: 20, lineHeight: 1.34, color: 'var(--osd-text)' }}>
      &ldquo;{response.excerpt}&rdquo;
    </p>
  </div>
);

const QualitativeCase = ({
  image,
  label,
  comparison,
  labelColor,
  delay,
}: {
  image: string;
  label: string;
  comparison: QualitativeComparison;
  labelColor: string;
  delay: number;
}) => (
  <div style={{ flex: 1, minWidth: 0, ...riseIn(delay) }}>
    <img
      src={image}
      style={{
        width: '100%',
        height: 236,
        objectFit: 'contain',
        display: 'block',
        background: '#f7ede3',
        borderRadius: 6,
      }}
      alt={label}
    />
    <div style={{ marginTop: 14, fontSize: 21, color: labelColor, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
      {label}
    </div>
    <ResponseExcerpt response={comparison.baseline} />
    <ResponseExcerpt response={comparison.guided} guided />
    <p style={{ margin: '12px 0 0', paddingTop: 10, borderTop: `1px solid ${muted}40`, fontSize: 21, lineHeight: 1.34, color: muted }}>
      {comparison.reading}
    </p>
  </div>
);

const QualitativeSuccess: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Results</Eyebrow>
    <PageHeading>What explanation guidance preserves</PageHeading>
    <div style={{ display: 'flex', gap: 64, marginTop: 44, alignItems: 'flex-start' }}>
      <div style={{ flex: 1.2, ...fadeIn(0.35) }}>
        <img src={qualBoleteImg} style={{ width: '100%', maxHeight: 560, objectFit: 'contain', borderRadius: 8 }} alt="Success patterns" />
      </div>
      <div style={{ flex: 0.9, paddingTop: 16 }}>
        <ul style={{ padding: 0, margin: 0 }}>
          <Bullet delay={0.45}><strong style={{ color: 'var(--osd-accent)' }}>Local texture & shape:</strong> guided students preserve cap texture, surface details, object structure more faithfully.</Bullet>
          <Bullet delay={0.58}><strong style={{ color: 'var(--osd-accent)' }}>Object identity recovery:</strong> explanation signal stabilizes grounding; bowls not shoes, animal not refusal.</Bullet>
          <Bullet delay={0.71}><strong style={{ color: 'var(--osd-accent)' }}>Generation robustness:</strong> baseline produces 67 no-image/refusal answers; guided variants produce almost none.</Bullet>
        </ul>
      </div>
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 12 — Qualitative limitations (image L + text R)
// ================================================================
const QualitativeLimits: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Results</Eyebrow>
    <PageHeading>Where the gap remains</PageHeading>
    <div style={{ display: 'flex', gap: 64, marginTop: 44, alignItems: 'flex-start' }}>
      <div style={{ flex: 1.2, ...fadeIn(0.35) }}>
        <img src={qualTriceratopsSceneImg} style={{ width: '100%', maxHeight: 560, objectFit: 'contain', borderRadius: 8 }} alt="Boundary patterns" />
      </div>
      <div style={{ flex: 0.9, paddingTop: 16 }}>
        <ul style={{ padding: 0, margin: 0 }}>
          <Bullet delay={0.45}><strong style={{ color: 'var(--osd-accent)' }}>Fine-grained semantics:</strong> dinosaur → &quot;generic skull&quot;; students miss specific categories, people, and scene context.</Bullet>
          <Bullet delay={0.58}><strong style={{ color: 'var(--osd-accent)' }}>Teacher still preferred</strong> on 350/500 images across every variant; the gap is real.</Bullet>
          <Bullet delay={0.71}><strong style={{ color: 'var(--osd-accent)' }}>Saliency is class-conditioned:</strong> the probe emphasizes discriminative evidence, not every detail needed for open-ended generation.</Bullet>
        </ul>
      </div>
    </div>
    <Footer />
  </div>
);

// ================================================================
// PAGE 11/12 - Slide-specific qualitative views with image-only assets
// ================================================================
const QualitativePatternSlide = ({
  title,
  image,
  imageAlt,
  comparison,
  positive,
}: {
  title: string;
  image: string;
  imageAlt: string;
  comparison: QualitativeComparison;
  positive: boolean;
}) => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Qualitative analysis</Eyebrow>
    <PageHeading>{title}</PageHeading>
    <div style={{ display: 'flex', gap: 64, marginTop: 42, alignItems: 'flex-start', maxWidth: 1480 }}>
      <img
        src={image}
        style={{ width: 500, height: 500, objectFit: 'contain', display: 'block', background: '#f7ede3', borderRadius: 6 }}
        alt={imageAlt}
      />
      <div style={{ flex: 1, minWidth: 0, paddingTop: 4 }}>
        <div style={{ fontSize: 21, fontWeight: 700, color: muted, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          {comparison.baseline.label}
        </div>
        <p style={{ margin: '10px 0 28px', fontSize: 28, lineHeight: 1.42, color: 'var(--osd-text)' }}>
          &ldquo;{comparison.baseline.excerpt}&rdquo;
        </p>
        <div style={{ fontSize: 21, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          {comparison.guided.label}
        </div>
        <p style={{ margin: '10px 0 30px', fontSize: 28, lineHeight: 1.42, color: 'var(--osd-text)' }}>
          &ldquo;{comparison.guided.excerpt}&rdquo;
        </p>
        <div style={{ borderTop: `2px solid ${positive ? amber : muted}55`, paddingTop: 18 }}>
          <div style={{ fontSize: 19, fontWeight: 700, color: positive ? 'var(--osd-accent)' : muted, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            Reading
          </div>
          <p style={{ margin: '8px 0 0', fontSize: 27, lineHeight: 1.42, color: muted }}>
            {comparison.reading}
          </p>
        </div>
      </div>
    </div>
    <Footer />
  </div>
);

const QualitativeSuccessSlide: Page = () => (
  <QualitativePatternSlide
    title="Saliency retains visible texture"
    image={qualBoleteImg}
    imageAlt="Mushroom with a raised reddish cap texture"
    comparison={qualitativeText.bolete}
    positive
  />
);

const ObjectIdentitySlide: Page = () => (
  <QualitativePatternSlide
    title="Guidance recovers object identity"
    image={qualMixingBowlImg}
    imageAlt="Three stacked red and brown containers"
    comparison={qualitativeText.mixing_bowl}
    positive
  />
);

const QualitativeLimitsSlide: Page = () => (
  <QualitativePatternSlide
    title="Full scene understanding limitations"
    image={qualTriceratopsSceneImg}
    imageAlt="Visitor next to a mounted Triceratops skull"
    comparison={qualitativeText.triceratops_scene}
    positive={false}
  />
);

const RobustnessSlide: Page = () => (
  <QualitativePatternSlide
    title="Guidance avoids a visual-path failure"
    image={qualTriceratopsSkeletonImg}
    imageAlt="Triceratops skeleton in an exhibition hall"
    comparison={qualitativeText.triceratops_skeleton}
    positive
  />
);

// ================================================================
// PAGE 13 — Limitations
// ================================================================
const Limitations: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Scope</Eyebrow>
    <PageHeading>Limitations</PageHeading>
    <ul style={{ marginTop: 52, marginBottom: 0, padding: 0, maxWidth: 1300 }}>
      <Bullet delay={0.3}><strong style={{ color: 'var(--osd-accent)' }}>Single VLM, single dataset:</strong>{' one teacher, Mini-ImageNet only.'}</Bullet>
      <Bullet delay={0.43}><strong style={{ color: 'var(--osd-accent)' }}>VLM judges, not humans:</strong>{' model-specific biases may exist.'}</Bullet>
      <Bullet delay={0.56}><strong style={{ color: 'var(--osd-accent)' }}>Saliency is probe-mediated:</strong>{' the probe\'s class-discriminative focus may not entirely align with the backbone.'}</Bullet>
      <Bullet delay={0.69}><strong style={{ color: 'var(--osd-accent)' }}>Teacher still dominates:</strong> the method improves fidelity but does not yet solve faithful visual-module replacement.</Bullet>
    </ul>
    <Footer />
  </div>
);

// ================================================================
// PAGE 14 — Conclusion
// ================================================================
const Conclusion: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: 120, position: 'relative' }}>
    <FlameBar />
    <Eyebrow>Conclusion</Eyebrow>
    <PageHeading>Explanations can be a useful<br />distillation signal</PageHeading>
    <ul style={{ marginTop: 52, marginBottom: 0, padding: 0, maxWidth: 1300 }}>
      <Bullet delay={0.3}>Grad-CAM-weighted token consistently outperforms plain global MSE<strong style={{ color: 'var(--osd-accent)' }}>{''}</strong>.</Bullet>
      <Bullet delay={0.45}>The strongest result comes from <strong style={{ color: 'var(--osd-accent)' }}>explanation-only supervision</strong>.</Bullet>
      <Bullet delay={0.6}>Gains manifest as <strong style={{ color: 'var(--osd-accent)' }}>preserved local evidence</strong> and reduced generation robustness failures.</Bullet>
      <Bullet delay={0.75}>Future work: richer prompts, human evaluation, stronger adapters, and <strong style={{ color: 'var(--osd-accent)' }}>saliency aligned with multimodal representations</strong>.</Bullet>
    </ul>
    <Footer />
  </div>
);

// ================================================================
// PAGE 15 — Thank you (with rocket)
// ================================================================
const Thanks: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '0 160px', position: 'relative', overflow: 'hidden' }}>
    <FlameBar />
    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: 800, height: 400, borderRadius: '50%', background: `radial-gradient(ellipse, ${amber}12, transparent 70%)`, animation: 'pulseGlow 4s ease-in-out infinite', pointerEvents: 'none' }} />
    <div style={{ textAlign: 'center', position: 'relative', zIndex: 1 }}>
      <div style={{ fontSize: 100, marginBottom: 30, ...riseIn(0.1) }}>🚀</div>
      <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 'var(--osd-size-hero)', fontWeight: 900, margin: 0, lineHeight: 1.08, ...riseIn(0.25) }}>Thank you</h2>
      <p style={{ fontSize: 40, color: muted, marginTop: 40, lineHeight: 1.5, ...riseIn(0.55) }}>Questions?</p>
      <div style={{ marginTop: 72, display: 'inline-block', height: 3, width: 200, background: `linear-gradient(90deg, ${amber}, ${crimson})`, ...fadeIn(0.8) }} />
    </div>
    <Footer />
  </div>
);

// === Transitions ===
export const transition: SlideTransition = {
  duration: 240,
  exit: { duration: 160, easing: EASE_IN, keyframes: [{ opacity: 1, transform: 'translateY(0)' }, { opacity: 0, transform: 'translateY(-6px)' }] },
  enter: { duration: 240, delay: 80, easing: EASE_OUT, keyframes: [{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'translateY(0)' }] },
};

Cover.transition = {
  duration: 300,
  exit: { duration: 160, easing: EASE_IN, keyframes: [{ opacity: 1, transform: 'translateY(0)' }, { opacity: 0, transform: 'translateY(-8px)' }] },
  enter: { duration: 300, delay: 100, easing: EASE_OUT, keyframes: [{ opacity: 0, transform: 'translateY(12px)', filter: 'blur(3px)' }, { opacity: 1, transform: 'translateY(0)', filter: 'blur(0)' }] },
};

// === Meta ===
export const meta: SlideMeta = {
  title: 'Look Where It Matters: Distilling Vision Through Explanations',
  createdAt: '2026-05-25T15:32:37.747Z',
};

export default [
  Cover,
  Motivation,
  Approach,
  Pipeline,
  TeacherProbes,
  GradCAM,
  Objectives,
  ProbeResults,
  PrimaryResults,
  QualitativeSuccessSlide,
  ObjectIdentitySlide,
  QualitativeLimitsSlide,
  RobustnessSlide,
  Limitations,
  Conclusion,
  Thanks,
] satisfies Page[];
