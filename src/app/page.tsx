'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import {
  BadgeCheck,
  Bot,
  Braces,
  Check,
  Copy,
  FileDown,
  Gauge,
  History,
  Loader2,
  ScanText,
  Sparkles,
  Split,
  Smile,
  Volume2,
  Wand2,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { AnalysisResult } from './api/analyze/route';

// ---------------------------------------------------------------------------
// Sample texts showcasing the different rule categories
// ---------------------------------------------------------------------------
const SAMPLES: { label: string; text: string }[] = [
  {
    label: 'Articles & agreement',
    text:
      'john and him goes to the market. he buyed two apple and a oranges. ' +
      'they was playing chess in an university while much people watched.',
  },
  {
    label: 'Word choice',
    text:
      'i dont believe that this product is more better then the old one. ' +
      'she have returned back the damaged goods without any receipt, but he said your welcome.',
  },
  {
    label: 'NER & spelling',
    text:
      'Mr. Robert Wilson is a honest man who works in New York. ' +
      'his company hired three engineer last year and it dont have much employees.',
  },
  {
    label: 'Clean paragraph',
    text:
      'Barack Obama was born in Hawaii in 1961. He served two terms as the ' +
      'President of the United States. The garden looks beautiful today.',
  },
];

// Badge color per grammar-rule category
const RULE_STYLES: Record<string, string> = {
  capitalization: 'bg-slate-100 text-slate-700 border-slate-200',
  'repeated-words': 'bg-rose-50 text-rose-700 border-rose-200',
  'irregular-past-tense': 'bg-amber-50 text-amber-800 border-amber-200',
  'subject-verb-agreement': 'bg-violet-50 text-violet-700 border-violet-200',
  'article-agreement': 'bg-teal-50 text-teal-700 border-teal-200',
  'number-noun-agreement': 'bg-lime-50 text-lime-700 border-lime-200',
  'confusion-pairs': 'bg-orange-50 text-orange-700 border-orange-200',
  'preposition-usage': 'bg-cyan-50 text-cyan-700 border-cyan-200',
  'missing-contraction': 'bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200',
  punctuation: 'bg-stone-100 text-stone-700 border-stone-200',
  'pronoun-case': 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'sentence-fragment': 'bg-red-50 text-red-700 border-red-200',
  spelling: 'bg-yellow-50 text-yellow-800 border-yellow-200',
};

// Badge label/color per style-suggestion type
const STYLE_STYLES: Record<string, { label: string; className: string }> = {
  'passive-voice': { label: 'Passive voice', className: 'bg-sky-50 text-sky-700 border-sky-200' },
  'long-sentence': { label: 'Long sentence', className: 'bg-purple-50 text-purple-700 border-purple-200' },
  'wordy-phrase': { label: 'Wordy phrase', className: 'bg-orange-50 text-orange-700 border-orange-200' },
  'adverb-overuse': { label: 'Adverb overuse', className: 'bg-pink-50 text-pink-700 border-pink-200' },
  'repeated-start': { label: 'Repetition', className: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
};

// ---------------------------------------------------------------------------
// Grammar-score ring (SVG donut used in the Writing Quality tab)
// ---------------------------------------------------------------------------
function ScoreRing({ score, label }: { score: number; label: string }) {
  const R = 34;
  const C = 2 * Math.PI * R;
  const color =
    score >= 90 ? '#059669' : score >= 75 ? '#16a34a' : score >= 60 ? '#d97706' : '#e11d48';
  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width="90"
        height="90"
        viewBox="0 0 90 90"
        role="img"
        aria-label={`Grammar score ${score} out of 100`}
      >
        <circle cx="45" cy="45" r={R} fill="none" stroke="#e5e7eb" strokeWidth="8" />
        <circle
          cx="45"
          cy="45"
          r={R}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - score / 100)}
          strokeLinecap="round"
          transform="rotate(-90 45 45)"
        />
        <text
          x="45"
          y="51"
          textAnchor="middle"
          fontSize="20"
          fontWeight="bold"
          className="fill-foreground"
        >
          {score}
        </text>
      </svg>
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs text-muted-foreground">grammar score</div>
    </div>
  );
}

const ENTITY_STYLES: Record<string, string> = {
  PERSON: 'bg-emerald-100 text-emerald-800',
  GPE: 'bg-violet-100 text-violet-800',
  ORGANIZATION: 'bg-amber-100 text-amber-800',
  LOCATION: 'bg-teal-100 text-teal-800',
  DATE: 'bg-orange-100 text-orange-800',
  MONEY: 'bg-lime-100 text-lime-800',
  PERCENT: 'bg-cyan-100 text-cyan-800',
  FACILITY: 'bg-stone-200 text-stone-800',
};

// ---------------------------------------------------------------------------
// Corrected text renderer: highlights every changed span in amber
// ---------------------------------------------------------------------------
function CorrectedTextView({ result }: { result: AnalysisResult }) {
  const text = result.correctedText;
  const spans = result.highlights
    .filter((h) => h.end > h.start)
    .sort((a, b) => a.start - b.start);

  const parts: { text: string; rule?: string }[] = [];
  let cursor = 0;
  for (const h of spans) {
    if (h.start > cursor) parts.push({ text: text.slice(cursor, h.start) });
    parts.push({ text: text.slice(h.start, h.end), rule: h.rule });
    cursor = h.end;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) });

  return (
    <p className="text-base leading-8 whitespace-pre-wrap">
      {parts.map((p, i) =>
        p.rule ? (
          <mark
            key={i}
            title={p.rule}
            className="bg-amber-200/70 text-amber-950 rounded px-0.5 font-medium"
          >
            {p.text}
          </mark>
        ) : (
          <span key={i}>{p.text}</span>
        )
      )}
    </p>
  );
}

// ---------------------------------------------------------------------------
// Analysis history (localStorage, last 8 analyses, click a chip to reload)
// ---------------------------------------------------------------------------
type HistoryItem = { text: string; ts: number; corrections: number };
const HISTORY_KEY = 'ai-grammar-studio-history';

function loadHistory(): HistoryItem[] {
  if (typeof window === 'undefined') return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function Home() {
  const [input, setInput] = useState(SAMPLES[0].text);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // load persisted history once on the client
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  function saveToHistory(text: string, corrections: number) {
    const next = loadHistory().filter((h) => h.text !== text);
    next.unshift({ text, ts: Date.now(), corrections });
    const trimmed = next.slice(0, 8);
    setHistory(trimmed);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
    } catch {
      /* storage unavailable — history is optional */
    }
  }

  function clearHistory() {
    setHistory([]);
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch {
      /* ignore */
    }
  }

  async function analyze() {
    if (!input.trim()) {
      toast.error('Please enter some text first.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      });
      const data = await res.json();
      if (!res.ok || data?.error) {
        setError(data?.error || 'Analysis failed. Please try again.');
      } else {
        const r = data as AnalysisResult;
        setResult(r);
        saveToHistory(input, r.corrections.length);
      }
    } catch {
      setError('Could not reach the analysis server. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  // ---------------------------- read the correction aloud -------------------
  function speak() {
    if (!result || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(result.correctedText);
    u.lang = 'en-US';
    u.rate = 1;
    window.speechSynthesis.speak(u);
  }

  // ----------------------------- download a full report ---------------------
  function downloadReport() {
    if (!result) return;
    const r = result;
    const lines = [
      'AI Grammar Studio — Analysis Report',
      `Generated: ${new Date().toLocaleString()}`,
      '',
      '--- ORIGINAL TEXT ---',
      input,
      '',
      '--- CORRECTED TEXT ---',
      r.correctedText,
      '',
      `--- CORRECTIONS (${r.corrections.length}) ---`,
      ...r.corrections.map(
        (c, i) =>
          `${i + 1}. [${c.ruleLabel}] ${c.original || '∅'} → ${
            c.suggestion || '(removed)'
          } — ${c.message}`
      ),
      '',
      '--- WRITING QUALITY ---',
      `Grammar score: ${r.quality.score}/100 (${r.quality.label})`,
      `Flesch Reading Ease: ${r.readability.fleschReadingEase} (${r.readability.label})`,
      `Flesch-Kincaid Grade Level: ${r.readability.fleschKincaidGrade}`,
      '',
      `--- STYLE SUGGESTIONS (${r.style.length}) ---`,
      ...(r.style.length
        ? r.style.map((s, i) => `${i + 1}. [${s.type}] ${s.message}`)
        : ['None']),
      '',
      '--- NAMED ENTITIES ---',
      r.entities.map((e) => `${e.text} (${e.type})`).join(', ') || 'None',
      '',
      '--- SENTIMENT (VADER) ---',
      `${r.sentiment.label} (compound ${r.sentiment.compound})`,
      '',
      '--- ML CLASSIFIER ---',
      `${r.classifier.label} (confidence ${(r.classifier.confidence * 100).toFixed(1)}%)`,
      '',
      '--- STATS ---',
      `Words: ${r.stats.wordCount} · Sentences: ${r.stats.sentenceCount} · Avg length: ${r.stats.avgSentenceLength} · Lexical diversity: ${r.stats.lexicalDiversity}`,
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `grammar-report-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Report downloaded.');
  }

  async function copyCorrected() {
    if (!result) return;
    await navigator.clipboard.writeText(result.correctedText);
    setCopied(true);
    toast.success('Corrected text copied to clipboard.');
    setTimeout(() => setCopied(false), 2000);
  }

  const sentimentEmoji =
    result?.sentiment.label === 'positive'
      ? '😊'
      : result?.sentiment.label === 'negative'
        ? '😞'
        : '😐';

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-emerald-50/60 via-background to-background">
      {/* ------------------------------ Header ------------------------------ */}
      <header className="border-b bg-background/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto flex items-center gap-3 px-4 py-4">
          <div className="h-10 w-10 rounded-xl bg-emerald-600 text-white flex items-center justify-center shadow-sm">
            <Wand2 className="h-5 w-5" aria-hidden />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-lg sm:text-xl font-bold tracking-tight leading-tight">
              AI Grammar Studio
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground">
              Tokenization · Stemming · Lemmatization · NER · Sentiment · Naive Bayes
            </p>
          </div>
          <Badge variant="outline" className="hidden sm:flex gap-1 items-center border-emerald-300 text-emerald-700">
            <Bot className="h-3.5 w-3.5" aria-hidden />
            NLTK + TextBlob
          </Badge>
        </div>
      </header>

      {/* ------------------------------- Main ------------------------------- */}
      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-6 space-y-6">
        {/* ------------------------- Input card ------------------------- */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ScanText className="h-4 w-4 text-emerald-600" aria-hidden />
              Your text
            </CardTitle>
            <CardDescription>
              Paste any English text — the engine finds grammar, tense,
              agreement, punctuation and spelling issues, then explains each fix.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={6}
              placeholder="Type or paste your text here…"
              aria-label="Text to analyze"
              className="text-base resize-y min-h-[140px]"
            />
            <div className="flex flex-wrap items-center gap-2">
              {SAMPLES.map((s) => (
                <button
                  key={s.label}
                  onClick={() => setInput(s.text)}
                  className="text-xs px-3 py-1.5 rounded-full border bg-background hover:bg-emerald-50 hover:border-emerald-300 transition-colors"
                >
                  {s.label}
                </button>
              ))}
            </div>
            {/* Analysis history (localStorage) */}
            {history.length > 0 && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <History className="h-3 w-3" aria-hidden />
                  Recent:
                </span>
                {history.map((h, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(h.text)}
                    title={`${h.text.slice(0, 80)} — ${h.corrections} corrections, ${new Date(
                      h.ts
                    ).toLocaleString()}`}
                    className="max-w-[200px] truncate text-xs px-3 py-1.5 rounded-full border bg-background hover:bg-emerald-50 hover:border-emerald-300 transition-colors"
                  >
                    {h.text.trim().slice(0, 40) || '(empty)'}
                  </button>
                ))}
                <button
                  onClick={clearHistory}
                  className="text-xs text-muted-foreground hover:text-rose-600 transition-colors"
                >
                  Clear
                </button>
              </div>
            )}
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <span className="text-xs text-muted-foreground">
                {input.length}/5000 characters
              </span>
              <Button
                onClick={analyze}
                disabled={loading}
                className="bg-emerald-600 hover:bg-emerald-700 min-h-11 px-6"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                    Analyzing…
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" aria-hidden />
                    Analyze &amp; Correct
                  </>
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              First analysis takes a few seconds while the Python NLP engine
              loads.
            </p>
          </CardContent>
        </Card>

        {/* --------------------------- Error ---------------------------- */}
        {error && (
          <Alert variant="destructive" role="alert">
            <AlertTitle>Analysis failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* ------------------------- Loading hint ----------------------- */}
        {loading && !result && (
          <div className="flex items-center justify-center gap-3 py-12 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin text-emerald-600" aria-hidden />
            <span className="text-sm">Running the NLP pipeline…</span>
          </div>
        )}

        {/* --------------------------- Results --------------------------- */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="space-y-6"
          >
            {/* Stat cards (incl. new grammar score) */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
              <Card className="p-4 flex flex-col items-center justify-center">
                <ScoreRing score={result.quality.score} label={result.quality.label} />
              </Card>
              <Card className="p-4">
                <div className="text-2xl font-bold">{result.corrections.length}</div>
                <div className="text-xs text-muted-foreground mt-1">corrections found</div>
              </Card>
              <Card className="p-4">
                <div className="text-2xl font-bold">{result.stats.wordCount}</div>
                <div className="text-xs text-muted-foreground mt-1">words</div>
              </Card>
              <Card className="p-4">
                <div className="text-2xl font-bold">{result.stats.sentenceCount}</div>
                <div className="text-xs text-muted-foreground mt-1">sentences</div>
              </Card>
              <Card className="p-4">
                <div className="text-2xl font-bold">{sentimentEmoji}</div>
                <div className="text-xs text-muted-foreground mt-1 capitalize">
                  {result.sentiment.label} tone
                </div>
              </Card>
            </div>

            {/* Corrected text */}
            <Card>
              <CardHeader className="flex-row items-start justify-between space-y-0">
                <div className="space-y-1.5">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Check className="h-4 w-4 text-emerald-600" aria-hidden />
                    Corrected text
                  </CardTitle>
                  <CardDescription>
                    Highlighted spans were changed by the engine.
                  </CardDescription>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={speak}
                    className="gap-1.5"
                    title="Read the corrected text aloud"
                  >
                    <Volume2 className="h-4 w-4" aria-hidden />
                    <span className="hidden sm:inline">Read aloud</span>
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadReport}
                    className="gap-1.5"
                    title="Download the full analysis report as a text file"
                  >
                    <FileDown className="h-4 w-4" aria-hidden />
                    <span className="hidden sm:inline">Report</span>
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={copyCorrected}
                    className="gap-1.5"
                  >
                    {copied ? <Check className="h-4 w-4" aria-hidden /> : <Copy className="h-4 w-4" aria-hidden />}
                    Copy
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <CorrectedTextView result={result} />
              </CardContent>
            </Card>

            {/* Analysis tabs */}
            <Card>
              <CardContent className="pt-6">
                <Tabs defaultValue="corrections">
                  <TabsList className="flex-wrap h-auto">
                    <TabsTrigger value="corrections" className="gap-1.5">
                      <BadgeCheck className="h-4 w-4" aria-hidden />
                      Corrections
                    </TabsTrigger>
                    <TabsTrigger value="pipeline" className="gap-1.5">
                      <Split className="h-4 w-4" aria-hidden />
                      NLP Pipeline
                    </TabsTrigger>
                    <TabsTrigger value="entities" className="gap-1.5">
                      <Braces className="h-4 w-4" aria-hidden />
                      Entities
                    </TabsTrigger>
                    <TabsTrigger value="sentiment" className="gap-1.5">
                      <Smile className="h-4 w-4" aria-hidden />
                      Sentiment
                    </TabsTrigger>
                    <TabsTrigger value="classifier" className="gap-1.5">
                      <Bot className="h-4 w-4" aria-hidden />
                      ML Classifier
                    </TabsTrigger>
                    <TabsTrigger value="quality" className="gap-1.5">
                      <Gauge className="h-4 w-4" aria-hidden />
                      Writing Quality
                    </TabsTrigger>
                  </TabsList>

                  {/* -------- Corrections -------- */}
                  <TabsContent value="corrections" className="mt-4 space-y-3">
                    {result.corrections.length === 0 && (
                      <div className="flex items-center gap-2 text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                        <BadgeCheck className="h-4 w-4" aria-hidden />
                        No grammar issues found — your text looks clean!
                      </div>
                    )}
                    {result.corrections.map((c, i) => (
                      <div
                        key={i}
                        className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 p-3 rounded-lg border bg-card hover:bg-muted/40 transition-colors"
                      >
                        <Badge
                          variant="outline"
                          className={`${RULE_STYLES[c.rule] ?? ''} shrink-0 w-fit`}
                        >
                          {c.ruleLabel}
                        </Badge>
                        <div className="flex-1 min-w-0 text-sm">
                          {c.suggestion === '' ? (
                            <>
                              <span className="line-through text-muted-foreground">
                                {c.original}
                              </span>
                              <span className="text-xs text-muted-foreground">
                                {' '}— removed
                              </span>
                            </>
                          ) : (
                            <>
                              <span className="line-through text-rose-600">{c.original || '∅'}</span>
                              <span className="mx-1.5 text-muted-foreground">→</span>
                              <span className="font-semibold text-emerald-700">{c.suggestion}</span>
                            </>
                          )}
                          <div className="text-xs text-muted-foreground mt-0.5">{c.message}</div>
                        </div>
                      </div>
                    ))}
                  </TabsContent>

                  {/* -------- NLP pipeline (tokenization/stemming/lemma) -------- */}
                  <TabsContent value="pipeline" className="mt-4">
                    <p className="text-sm text-muted-foreground mb-3">
                      Each word tokenized with NLTK, stemmed with the Porter
                      stemmer and lemmatized with WordNet (POS-aware).
                      {result.pipeline.truncated && ' Showing the first 120 tokens.'}
                    </p>
                    <div className="max-h-96 overflow-y-auto rounded-lg border">
                      <Table>
                        <TableHeader className="sticky top-0 bg-background">
                          <TableRow>
                            <TableHead>Token</TableHead>
                            <TableHead>POS tag</TableHead>
                            <TableHead>Porter stem</TableHead>
                            <TableHead>WordNet lemma</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.pipeline.tokens.map((t, i) => (
                            <TableRow key={i}>
                              <TableCell className="font-medium">{t.token}</TableCell>
                              <TableCell>
                                <Badge variant="secondary" className="font-mono text-xs">
                                  {t.pos}
                                </Badge>
                              </TableCell>
                              <TableCell className="font-mono text-xs text-muted-foreground">
                                {t.stem}
                              </TableCell>
                              <TableCell className="font-mono text-xs text-emerald-700">
                                {t.lemma}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </TabsContent>

                  {/* -------- Entities (NER) -------- */}
                  <TabsContent value="entities" className="mt-4">
                    {result.entities.length === 0 ? (
                      <p className="text-sm text-muted-foreground p-4 border rounded-lg">
                        No named entities detected in this text.
                      </p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {result.entities.map((e, i) => (
                          <span
                            key={i}
                            className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm border ${
                              ENTITY_STYLES[e.type] ?? 'bg-muted'
                            }`}
                          >
                            <span className="font-semibold">{e.text}</span>
                            <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                              {e.type}
                            </Badge>
                          </span>
                        ))}
                      </div>
                    )}
                    <p className="text-xs text-muted-foreground mt-4">
                      Detected with NLTK&apos;s <code>ne_chunk</code> named-entity
                      chunker (PERSON, GPE, ORGANIZATION, …).
                    </p>
                  </TabsContent>

                  {/* -------- Sentiment -------- */}
                  <TabsContent value="sentiment" className="mt-4 space-y-4">
                    <div className="flex items-center gap-3">
                      <span className="text-4xl">{sentimentEmoji}</span>
                      <div>
                        <div className="text-lg font-semibold capitalize">
                          {result.sentiment.label}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          VADER compound score: {result.sentiment.compound} (−1 to +1)
                        </div>
                      </div>
                    </div>
                    <Separator />
                    {(
                      [
                        ['Positive', result.sentiment.positive, 'bg-emerald-500'],
                        ['Neutral', result.sentiment.neutral, 'bg-stone-400'],
                        ['Negative', result.sentiment.negative, 'bg-rose-500'],
                      ] as const
                    ).map(([label, val, color]) => (
                      <div key={label} className="space-y-1.5">
                        <div className="flex justify-between text-sm">
                          <span>{label}</span>
                          <span className="text-muted-foreground">{val}</span>
                        </div>
                        <div className="h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className={`h-full rounded-full ${color} transition-all`}
                            style={{ width: `${Math.round(val * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </TabsContent>

                  {/* -------- ML classifier -------- */}
                  <TabsContent value="classifier" className="mt-4 space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                      <div
                        className={`rounded-xl px-5 py-4 border ${
                          result.classifier.label === 'grammatical'
                            ? 'bg-emerald-50 border-emerald-200'
                            : 'bg-amber-50 border-amber-200'
                        }`}
                      >
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                          Model verdict
                        </div>
                        <div
                          className={`text-xl font-bold capitalize ${
                            result.classifier.label === 'grammatical'
                              ? 'text-emerald-700'
                              : 'text-amber-700'
                          }`}
                        >
                          {result.classifier.label}
                        </div>
                      </div>
                      <div className="flex-1 space-y-1.5">
                        <div className="flex justify-between text-sm">
                          <span>Confidence</span>
                          <span className="text-muted-foreground">
                            {(result.classifier.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <Progress
                          value={result.classifier.confidence * 100}
                          className="h-2"
                        />
                      </div>
                    </div>
                    <Separator />
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="font-semibold">
                          {((result.classifier.accuracy ?? 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-muted-foreground">hold-out accuracy</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="font-semibold">{result.classifier.trainSize ?? '—'}</div>
                        <div className="text-xs text-muted-foreground">training sentences</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <div className="font-semibold">{result.classifier.testSize ?? '—'}</div>
                        <div className="text-xs text-muted-foreground">test sentences</div>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Multinomial Naive Bayes (NLTK) trained on Brown-corpus
                      sentences vs. programmatically corrupted versions, using
                      POS-ratio, length and error-pattern features.
                    </p>
                  </TabsContent>

                  {/* -------- Writing quality (score + readability + style) -------- */}
                  <TabsContent value="quality" className="mt-4 space-y-5">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div className="flex items-center justify-center p-4 rounded-lg border bg-muted/30">
                        <ScoreRing score={result.quality.score} label={result.quality.label} />
                      </div>
                      <div className="p-4 rounded-lg border bg-muted/30 space-y-1.5">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                          Flesch Reading Ease
                        </div>
                        <div className="text-3xl font-bold">
                          {result.readability.fleschReadingEase}
                        </div>
                        <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200">
                          {result.readability.label}
                        </Badge>
                        <p className="text-xs text-muted-foreground">
                          0–100 — higher means easier to read.
                        </p>
                      </div>
                      <div className="p-4 rounded-lg border bg-muted/30 space-y-1.5">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                          Grade level
                        </div>
                        <div className="text-3xl font-bold">
                          {result.readability.fleschKincaidGrade}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Flesch–Kincaid Grade Level — the approx. US school
                          grade needed to read this text.
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {result.readability.syllables} syllables in total.
                        </p>
                      </div>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <div className="text-sm font-medium">
                        Style suggestions ({result.style.length})
                      </div>
                      {result.style.length === 0 ? (
                        <div className="flex items-center gap-2 text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                          <Gauge className="h-4 w-4" aria-hidden />
                          No style issues detected — nice and readable.
                        </div>
                      ) : (
                        result.style.map((s, i) => (
                          <div
                            key={i}
                            className="p-3 rounded-lg border bg-card hover:bg-muted/40 transition-colors"
                          >
                            <Badge
                              variant="outline"
                              className={`${STYLE_STYLES[s.type]?.className ?? ''} w-fit`}
                            >
                              {STYLE_STYLES[s.type]?.label ?? s.type}
                            </Badge>
                            <p className="text-sm mt-1.5">{s.message}</p>
                            {s.match && (
                              <p className="text-xs text-muted-foreground mt-0.5 font-mono">
                                matched: {s.match}
                              </p>
                            )}
                          </div>
                        ))
                      )}
                      <p className="text-xs text-muted-foreground">
                        Style hints are advisory only — the engine never
                        rewrites your voice automatically.
                      </p>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </main>

      {/* ------------------------------ Footer ------------------------------ */}
      <footer className="mt-auto border-t bg-background">
        <div className="max-w-5xl mx-auto px-4 py-4 text-xs text-muted-foreground flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            AI Grammar Studio — built with Next.js, NLTK, TextBlob and a Naive
            Bayes classifier.
          </span>
          <span>Pipeline: tokenize → stem → lemmatize → correct → classify → NER → sentiment</span>
        </div>
      </footer>
    </div>
  );
}
