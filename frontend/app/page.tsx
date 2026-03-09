"use client";

import Link from "next/link";
import styles from "./landing.module.css";
import {
  motion,
  useInView,
  useMotionValue,
  useSpring,
  useTransform,
} from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";

const stats = [
  { value: 50000, suffix: "+", label: "Assignments processed" },
  { value: 98, suffix: "%", label: "Teacher satisfaction" },
  { value: 2, prefix: "< ", suffix: " min", label: "Avg. turnaround time" },
  { value: 4.9, suffix: "/5", label: "Average rating" },
];

const steps = [
  {
    number: "1",
    title: "Upload",
    text: "Submit PDF, DOCX, TXT, tables, images, or audio through one secure workflow.",
  },
  {
    number: "2",
    title: "Ingestion",
    text: "Scanning, extraction, OCR, table parsing, and transcript generation happen automatically.",
  },
  {
    number: "3",
    title: "AI Feedback",
    text: "ML signals and LLM reasoning generate structured academic guidance with confidence awareness.",
  },
  {
    number: "4",
    title: "Review",
    text: "Students improve drafts while professors review rubric-aligned outputs more consistently.",
  },
];

const audienceCards = [
  {
    title: "For Students",
    points: [
      "Clear feedback that explains strengths and weaknesses",
      "Actionable next steps instead of vague comments",
      "Fast turnaround for drafts and resubmissions",
      "Confidence-aware insights for better revision",
    ],
  },
  {
    title: "For Educators",
    points: [
      "Rubric-aligned assessment support",
      "More consistent academic review workflows",
      "Evidence-backed explanations for decisions",
      "Time-saving moderation and report visibility",
    ],
  },
];

const testimonials = [
  {
    quote:
      "The platform makes feedback feel immediate, structured, and much easier for students to act on.",
    name: "Academic Reviewer",
    role: "Demo feedback sample",
  },
  {
    quote:
      "The rubric-style breakdown is what makes it useful. It looks far more practical than a generic AI response.",
    name: "Professor User",
    role: "Demo feedback sample",
  },
  {
    quote:
      "It gives students something they can actually improve from, while still helping staff review faster.",
    name: "Teaching Team",
    role: "Demo feedback sample",
  },
];

type PreviewMode = "student" | "professor" | "admin";

function BrandIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="brandGrad" x1="10" y1="8" x2="54" y2="56" gradientUnits="userSpaceOnUse">
          <stop stopColor="#5B5CF0" />
          <stop offset="1" stopColor="#7C3AED" />
        </linearGradient>
      </defs>

      <rect x="6" y="6" width="52" height="52" rx="16" fill="url(#brandGrad)" />
      <path
        d="M20 24.5L32 18L44 24.5L32 31L20 24.5Z"
        fill="white"
        fillOpacity="0.98"
      />
      <path
        d="M24 30.5V37.8C24 40.2 27.7 43 32 43C36.3 43 40 40.2 40 37.8V30.5L32 35L24 30.5Z"
        fill="white"
        fillOpacity="0.98"
      />
      <path
        d="M44 24.5V33"
        stroke="white"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <circle cx="44" cy="35.5" r="2.2" fill="white" />
      <path
        d="M47.7 17.8L49 20.6L52 21.1L49.8 23.2L50.3 26.2L47.7 24.8L45 26.2L45.5 23.2L43.4 21.1L46.3 20.6L47.7 17.8Z"
        fill="#E9D5FF"
      />
    </svg>
  );
}

function CountUpStat({
  value,
  label,
  prefix = "",
  suffix = "",
}: {
  value: number;
  label: string;
  prefix?: string;
  suffix?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!isInView) return;

    const duration = 1400;
    const start = performance.now();

    const animate = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = value * eased;
      setDisplay(current);
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [isInView, value]);

  const formatted = useMemo(() => {
    if (value % 1 !== 0) return display.toFixed(1);
    if (value >= 1000) return Math.round(display).toLocaleString();
    return Math.round(display).toString();
  }, [display, value]);

  return (
    <div ref={ref} className={styles.stat}>
      <b>
        {prefix}
        {formatted}
        {suffix}
      </b>
      <span>{label}</span>
    </div>
  );
}

function Reveal({
  children,
  delay = 0,
  y = 28,
}: {
  children: React.ReactNode;
  delay?: number;
  y?: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y }}
      transition={{ duration: 0.65, ease: "easeOut", delay }}
    >
      {children}
    </motion.div>
  );
}

function HeroPreview() {
  const [mode, setMode] = useState<PreviewMode>("professor");
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const rotateX = useSpring(useTransform(y, [-80, 80], [5, -5]), {
    stiffness: 160,
    damping: 18,
  });
  const rotateY = useSpring(useTransform(x, [-80, 80], [-5, 5]), {
    stiffness: 160,
    damping: 18,
  });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    x.set(px - cx);
    y.set(py - cy);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      ref={wrapRef}
      className={styles.previewWrap}
      style={{ rotateX, rotateY, transformPerspective: 1200 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      initial={{ opacity: 0, y: 24, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.75, ease: "easeOut", delay: 0.2 }}
    >
      <div className={styles.previewGlow} />
      <div className={styles.previewGlowTwo} />

      <div className={styles.previewTop}>
        <div className={styles.previewDots}>
          <span />
          <span />
          <span />
        </div>
        <div className={styles.previewTag}>Live Preview</div>
      </div>

      <div className={styles.previewBody}>
        <aside className={styles.previewSidebar}>
          <div>
            <div className={styles.previewSidebarLogo}>
              <div className={styles.previewSidebarLogoIconWrap}>
                <BrandIcon className={styles.previewSidebarLogoIcon} />
              </div>
              <div>
                <strong>EduAI</strong>
                <span>Assessment OS</span>
              </div>
            </div>

            <div className={styles.sideGroup}>
              <div className={mode === "student" ? styles.sideItemActive : styles.sideItem}>
                Student View
              </div>
              <div className={mode === "professor" ? styles.sideItemActive : styles.sideItem}>
                Professor View
              </div>
              <div className={mode === "admin" ? styles.sideItemActive : styles.sideItem}>
                Admin View
              </div>
            </div>
          </div>

          <div className={styles.previewSidebarFooter}>
            <div className={styles.sideItem}>Uploads</div>
            <div className={styles.sideItem}>Analytics</div>
          </div>
        </aside>

        <div className={styles.previewMain}>
          <div className={styles.previewTabs}>
            <button
              className={mode === "student" ? styles.previewTabActive : styles.previewTab}
              onClick={() => setMode("student")}
              type="button"
            >
              Student
            </button>
            <button
              className={mode === "professor" ? styles.previewTabActive : styles.previewTab}
              onClick={() => setMode("professor")}
              type="button"
            >
              Professor
            </button>
            <button
              className={mode === "admin" ? styles.previewTabActive : styles.previewTab}
              onClick={() => setMode("admin")}
              type="button"
            >
              Admin
            </button>
          </div>

          {mode === "student" && (
            <motion.div
              key="student"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
            >
              <div className={styles.previewHeaderRow}>
                <div>
                  <h4>Draft Improvement View</h4>
                  <p>Personalised feedback with revision priorities</p>
                </div>
                <div className={styles.liveBadge}>Student Mode</div>
              </div>

              <div className={styles.topAnalytics}>
                <div className={styles.heroMetric}>
                  <span>Clarity</span>
                  <strong>88%</strong>
                </div>
                <div className={styles.heroMetric}>
                  <span>Structure</span>
                  <strong>91%</strong>
                </div>
                <div className={styles.heroMetric}>
                  <span>Confidence</span>
                  <strong>92%</strong>
                </div>
              </div>

              <div className={styles.progressPanel}>
                <div className={styles.barLabel}>
                  <span>Revision completion</span>
                  <span>72%</span>
                </div>
                <div className={styles.progressBar}>
                  <div className={`${styles.progressFill} ${styles.progressFillStudent}`} />
                </div>

                <div className={styles.stepPills}>
                  <span className={styles.stepDone}>Upload</span>
                  <span className={styles.stepDone}>Extract</span>
                  <span className={styles.stepActive}>Feedback</span>
                  <span className={styles.stepPending}>Resubmit</span>
                </div>
              </div>

              <div className={styles.previewGrid}>
                <div className={styles.reportBox}>
                  <div className={styles.reportBoxHead}>
                    <strong>Actionable Next Steps</strong>
                    <span>Student summary</span>
                  </div>

                  <div className={styles.lineLong} />
                  <div className={styles.lineMid} />
                  <div className={styles.lineShort} />

                  <div className={styles.rubricRows}>
                    <div className={styles.rubricRow}>
                      <span>Strengthen argument depth</span>
                      <b>High</b>
                    </div>
                    <div className={styles.rubricRow}>
                      <span>Add stronger evidence</span>
                      <b>Medium</b>
                    </div>
                    <div className={styles.rubricRow}>
                      <span>Improve structure flow</span>
                      <b>High</b>
                    </div>
                  </div>
                </div>

                <div className={styles.sideInsightPanel}>
                  <div className={styles.sideInsightCard}>
                    <span>Focus Area</span>
                    <strong>Critical analysis first</strong>
                    <p>Most impactful improvement area for the next submission.</p>
                  </div>
                  <div className={styles.sideInsightCard}>
                    <span>AI Guidance</span>
                    <strong>Clear revision path</strong>
                    <p>Feedback is broken into concrete, understandable steps.</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {mode === "professor" && (
            <motion.div
              key="professor"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
            >
              <div className={styles.previewHeaderRow}>
                <div>
                  <h4>Assignment Review</h4>
                  <p>Upload → Extraction → AI Review → Report</p>
                </div>
                <div className={styles.liveBadge}>Professor View</div>
              </div>

              <div className={styles.topAnalytics}>
                <div className={styles.heroMetric}>
                  <span>Clarity</span>
                  <strong>88%</strong>
                </div>
                <div className={styles.heroMetric}>
                  <span>Structure</span>
                  <strong>91%</strong>
                </div>
                <div className={styles.heroMetric}>
                  <span>Rubric Match</span>
                  <strong>94%</strong>
                </div>
              </div>

              <div className={styles.progressPanel}>
                <div className={styles.barLabel}>
                  <span>Processing pipeline</span>
                  <span>87%</span>
                </div>
                <div className={styles.progressBar}>
                  <div className={styles.progressFill} />
                </div>

                <div className={styles.stepPills}>
                  <span className={styles.stepDone}>Upload</span>
                  <span className={styles.stepDone}>Extract</span>
                  <span className={styles.stepActive}>AI Review</span>
                  <span className={styles.stepPending}>Report</span>
                </div>
              </div>

              <div className={styles.previewGrid}>
                <div className={styles.reportBox}>
                  <div className={styles.reportBoxHead}>
                    <strong>Feedback Summary</strong>
                    <span>Professor mode</span>
                  </div>

                  <div className={styles.lineLong} />
                  <div className={styles.lineMid} />
                  <div className={styles.lineShort} />

                  <div className={styles.rubricRows}>
                    <div className={styles.rubricRow}>
                      <span>Critical Analysis</span>
                      <b>High</b>
                    </div>
                    <div className={styles.rubricRow}>
                      <span>Evidence Use</span>
                      <b>Medium</b>
                    </div>
                    <div className={styles.rubricRow}>
                      <span>Academic Structure</span>
                      <b>High</b>
                    </div>
                  </div>
                </div>

                <div className={styles.sideInsightPanel}>
                  <div className={styles.sideInsightCard}>
                    <span>Rubric Lens</span>
                    <strong>Aligned review logic</strong>
                    <p>Feedback is structured around academic criteria and marking consistency.</p>
                  </div>
                  <div className={styles.sideInsightCard}>
                    <span>AI Confidence</span>
                    <strong>92% stable output</strong>
                    <p>Reasoning is confidence-aware and suitable for moderation support.</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {mode === "admin" && (
            <motion.div
              key="admin"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
            >
              <div className={styles.previewHeaderRow}>
                <div>
                  <h4>System Overview</h4>
                  <p>Operational visibility across reports, models, and alerts</p>
                </div>
                <div className={styles.liveBadge}>Admin View</div>
              </div>

              <div className={styles.topAnalytics}>
                <div className={styles.heroMetric}>
                  <span>AI Runs</span>
                  <strong>12.4k</strong>
                </div>
                <div className={styles.heroMetric}>
                  <span>Active Users</span>
                  <strong>2.1k</strong>
                </div>
                <div className={styles.heroMetric}>
                  <span>Alerts</span>
                  <strong>06</strong>
                </div>
              </div>

              <div className={styles.progressPanel}>
                <div className={styles.barLabel}>
                  <span>Infrastructure health</span>
                  <span>96%</span>
                </div>
                <div className={styles.progressBar}>
                  <div className={`${styles.progressFill} ${styles.progressFillAdmin}`} />
                </div>

                <div className={styles.stepPills}>
                  <span className={styles.stepDone}>Workers</span>
                  <span className={styles.stepDone}>Models</span>
                  <span className={styles.stepActive}>Audit</span>
                  <span className={styles.stepPending}>Incidents</span>
                </div>
              </div>

              <div className={styles.previewGrid}>
                <div className={styles.reportBox}>
                  <div className={styles.reportBoxHead}>
                    <strong>Operational Snapshot</strong>
                    <span>System panel</span>
                  </div>

                  <div className={styles.rubricRows}>
                    <div className={styles.rubricRow}>
                      <span>Worker queue status</span>
                      <b>Healthy</b>
                    </div>
                    <div className={styles.rubricRow}>
                      <span>Model registry sync</span>
                      <b>Stable</b>
                    </div>
                    <div className={styles.rubricRow}>
                      <span>Security events</span>
                      <b>Monitored</b>
                    </div>
                  </div>
                </div>

                <div className={styles.sideInsightPanel}>
                  <div className={styles.sideInsightCard}>
                    <span>Audit Logs</span>
                    <strong>Full activity tracking</strong>
                    <p>Uploads, AI calls, failures, and review events are visible.</p>
                  </div>
                  <div className={styles.sideInsightCard}>
                    <span>Security</span>
                    <strong>Prompt threats flagged</strong>
                    <p>Suspicious content and pipeline issues surface early.</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      <div className={styles.floatCardLeft}>
        <span>Student Experience</span>
        <strong>Revision guidance in seconds</strong>
      </div>

      <div className={styles.floatCardRight}>
        <span>Professor Experience</span>
        <strong>Consistent rubric-based review</strong>
      </div>
    </motion.div>
  );
}

export default function LandingPage() {
  return (
    <main className={styles.page}>
      <div className={styles.bgGlowOne} />
      <div className={styles.bgGlowTwo} />
      <div className={styles.bgMesh} />
      <div className={styles.bgGrid} />

      <header className={styles.header}>
        <div className={styles.container}>
          <motion.div
            className={styles.nav}
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: "easeOut" }}
          >
            <div className={styles.brand}>
              <div className={styles.brandMark}>
                <BrandIcon className={styles.brandSvg} />
              </div>
              <div className={styles.brandTextWrap}>
                <span className={styles.brandTitle}>EduFeedback AI</span>
                <span className={styles.brandSub}>Academic Intelligence Platform</span>
              </div>
            </div>

            <nav className={styles.navLinks}>
              <a href="#how">How it works</a>
              <a href="#audience">Built for everyone</a>
              <a href="#proof">Proof</a>
            </nav>

            <div className={styles.navActions}>
              <Link className={styles.btn} href="/login">
                Login
              </Link>
              <Link className={styles.btnPrimary} href="/register">
                Start Free Trial
              </Link>
            </div>
          </motion.div>
        </div>
      </header>

      <section className={styles.hero}>
        <div className={styles.container}>
          <div className={styles.heroGrid}>
            <motion.div
              className={styles.heroLeft}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.75, ease: "easeOut" }}
            >
              <motion.span
                className={styles.badge}
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.45, delay: 0.1 }}
              >
                AI Powered Academic Assessment
              </motion.span>

              <motion.h1
                className={styles.heroH1}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.12 }}
              >
                Smarter Feedback.
                <br />
                Better Academic Review.
              </motion.h1>

              <motion.p
                className={styles.heroP}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.18 }}
              >
                Upload assignments, generate structured feedback, and help both
                students and professors move through assessment with more clarity,
                consistency, and speed.
              </motion.p>

              <motion.div
                className={styles.heroBtns}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.24 }}
              >
                <Link className={styles.btnPrimaryLg} href="/register">
                  Start Free Trial
                </Link>
                <a className={styles.btnLg} href="#demo">
                  Watch Demo
                </a>
              </motion.div>

              <motion.div
                className={styles.statsRow}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                {stats.map((item) => (
                  <CountUpStat
                    key={item.label}
                    value={item.value}
                    prefix={item.prefix}
                    suffix={item.suffix}
                    label={item.label}
                  />
                ))}
              </motion.div>
            </motion.div>

            <div id="demo" className={styles.heroPreviewCol}>
              <HeroPreview />
            </div>
          </div>
        </div>
      </section>

      <section className={styles.logoSection} id="proof">
        <div className={styles.container}>
          <Reveal>
            <p className={styles.logoHeading}>Built for modern academic workflows</p>
            <div className={styles.logoRow}>
              <div className={styles.logoChip}>EduFeedback AI</div>
              <div className={styles.logoChip}>University Partner</div>
              <div className={styles.logoChip}>Faculty Demo</div>
              <div className={styles.logoChip}>Assessment Lab</div>
              <div className={styles.logoChip}>Research Pilot</div>
            </div>
          </Reveal>
        </div>
      </section>

      <section className={styles.audienceSection} id="audience">
        <div className={styles.container}>
          <Reveal>
            <div className={styles.sectionHead}>
              <span className={styles.sectionBadge}>Built for Everyone</span>
              <h2>Designed for both students and educators</h2>
              <p>
                This platform supports two connected experiences: student
                improvement and professor review quality.
              </p>
            </div>
          </Reveal>

          <div className={styles.audienceGrid}>
            {audienceCards.map((card, index) => (
              <Reveal key={card.title} delay={index * 0.12}>
                <div className={styles.audienceCard}>
                  <h3>{card.title}</h3>
                  <ul>
                    {card.points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.hiw} id="how">
        <div className={styles.container}>
          <Reveal>
            <div className={styles.sectionHead}>
              <span className={styles.sectionBadge}>How It Works</span>
              <h2>From upload to rubric-mapped feedback</h2>
              <p>A simple but polished academic assessment pipeline.</p>
            </div>
          </Reveal>

          <div className={styles.hiwGrid}>
            {steps.map((step, index) => (
              <Reveal key={step.number} delay={index * 0.08}>
                <div className={styles.hiwCard}>
                  <div className={styles.iconDot}>{step.number}</div>
                  <h4>{step.title}</h4>
                  <p>{step.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.testimonialSection}>
        <div className={styles.container}>
          <Reveal>
            <div className={styles.sectionHead}>
              <span className={styles.sectionBadge}>Feedback Highlights</span>
              <h2>Demo comments you can show on the landing page</h2>
              <p>
                These are polished demo testimonials. Replace them with real user
                quotes once you collect them.
              </p>
            </div>
          </Reveal>

          <div className={styles.testimonialGrid}>
            {testimonials.map((item, index) => (
              <Reveal key={item.quote} delay={index * 0.1}>
                <div className={styles.testimonialCard}>
                  <p className={styles.quote}>“{item.quote}”</p>
                  <div className={styles.testimonialMeta}>
                    <strong>{item.name}</strong>
                    <span>{item.role}</span>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.cta}>
        <div className={styles.container}>
          <Reveal>
            <div className={styles.ctaBox}>
              <div>
                <h3>Ready to Transform Assessment?</h3>
                <p>
                  Join universities and academic teams using AI to enhance
                  feedback, review quality, and student outcomes.
                </p>
              </div>

              <div className={styles.ctaBtns}>
                <Link className={styles.ctaBtnPrimary} href="/register">
                  Get Started Now
                </Link>
                <Link className={styles.ctaBtnSecondary} href="/login">
                  Explore Login
                </Link>
              </div>
            </div>
          </Reveal>
        </div>
      </section>
    </main>
  );
}