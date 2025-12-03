import React, { useEffect, useRef, useState } from 'react';
import { Upload, Send, Activity, Loader2, CheckCircle2, BoxSelect, X } from 'lucide-react';

/**
 * API base: set VITE_API_URL in env; if empty, use relative paths (Netlify proxy).
 */
const envBase = (import.meta.env.VITE_API_URL as string | undefined)?.trim() || '';
const API_BASE = envBase ? envBase.replace(/\/$/, '') : '';

type MatchedElement = {
  page: number;
  bbox: number[];
  text: string;
  image_url?: string;
  page_width_px?: number;
  page_height_px?: number;
  page_width_pdf?: number;
  page_height_pdf?: number;
  bbox_norm?: number[];
  page_bbox_max_x_px?: number | null;
  page_bbox_max_y_px?: number | null;
};

type QueryResponse = {
  question: string;
  answer_summary: string;
  reasoning: string;
  matched_elements: MatchedElement[];
};

type RawElement = {
  type: string;
  content: string;
  page_number?: number;
  bbox_pixel?: number[] | null;
  bbox_pdf?: number[] | null;
  confidence?: number | null;
  needs_review?: boolean;
};

type SectionContent = {
  id?: string;
  title?: string;
  type?: string;
  page_range?: number[];
  page_count?: number;
  confidence?: number | null;
  pages?: Array<{
    page_number?: number;
    elements?: RawElement[];
  }>;
};

type PageSnapshot = {
  page_number?: number;
  elements?: RawElement[];
};

type Medication = {
  drug_name: string;
  dosage: string | null;
  frequency: string | null;
  start_date: string | null;
  end_date: string | null;
  notes: string | null;
};

type Lab = {
  test_name: string;
  date_ordered: string | null;
  result: string | null;
  reason: string | null;
  notes: string | null;
};

type Procedure = {
  procedure_name: string;
  date: string | null;
  purpose: string | null;
  result: string | null;
  notes: string | null;
};

type StructuredData = {
  summary: string | null;
  medications: Medication[];
  labs: Lab[];
  procedures: Procedure[];
  patient_demographics?: RawElement | null;
  sections?: SectionContent[];
  pages?: PageSnapshot[];
  document_dates?: string[] | null;
};

type ExtractionResponse = {
  document_id: string;
  extractions: {
    full: string;
    enhanced: string;
    index: string;
  };
};

type DocumentRecord = {
  id: string;
  original_filename?: string;
  storage_url?: string;
  created_at?: string;
};

type DocumentResponse = {
  document: DocumentRecord;
  extractions: Array<{ kind: string; storage_url: string; created_at?: string }>;
};

type RecentDoc = {
  id: string;
  name?: string;
  createdAt?: string;
};

type RecentQuery = {
  docId: string;
  question: string;
  answer: string;
  reasoning?: string;
  evidence?: MatchedElement[];
};

const RECENT_DOCS_KEY = 'ehrx_recent_docs';
const RECENT_QUERIES_KEY = 'ehrx_recent_queries';

const loadLocal = <T,>(key: string, fallback: T): T => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
};

const saveLocal = (key: string, value: unknown) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
};

const api = {
  upload: async (file: File): Promise<{ document_id: string }> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/documents`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Upload failed');
    }
    return res.json();
  },

  extract: async (docId: string): Promise<ExtractionResponse> => {
    const res = await fetch(`${API_BASE}/documents/${docId}/extract`, {
      method: 'POST',
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Extraction failed');
    }
    return res.json();
  },

  query: async (docId: string, question: string): Promise<QueryResponse> => {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_id: docId, question, kind: 'enhanced' }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Query failed');
    }
    return res.json();
  },

  getDocument: async (docId: string): Promise<DocumentResponse> => {
    const res = await fetch(`${API_BASE}/documents/${docId}`, { method: 'GET' });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Document lookup failed');
    }
    return res.json();
  },

  getStructuredData: async (docId: string): Promise<StructuredData> => {
    const res = await fetch(`${API_BASE}/api/documents/${docId}/structured-data`, { method: 'GET' });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Failed to fetch structured data');
    }
    return res.json();
  },
};

const Header = () => (
  <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-md">
    <div className="flex items-center gap-3">
      <div>
        <h1 className="text-lg font-semibold text-slate-100 tracking-tight leading-none">EHRX</h1>
      </div>
    </div>
    <div className="flex items-center gap-3 bg-slate-800 rounded-full px-4 py-2 border border-slate-700 text-xs text-slate-200">
      <span className="text-emerald-300 font-semibold">Live API</span>
      <span className="text-slate-400 truncate max-w-[200px]">{API_BASE || 'proxied'}</span>
    </div>
  </header>
);

const UploadCard = ({ onUpload }: { onUpload: (f: File) => void }) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) onUpload(e.dataTransfer.files[0]);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`relative group overflow-hidden border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 ${
        isDragging
          ? 'border-blue-500 bg-blue-50 scale-[1.02]'
          : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50 bg-white shadow-sm hover:shadow-md'
      }`}
    >
      <input
        type="file"
        ref={inputRef}
        className="hidden"
        accept=".pdf"
        onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
      />

      <div
        className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 transition-colors duration-300 ${
          isDragging ? 'bg-blue-100' : 'bg-slate-100 group-hover:bg-blue-50'
        }`}
      >
        <Upload className={`w-8 h-8 ${isDragging ? 'text-blue-600' : 'text-slate-400 group-hover:text-blue-500'}`} />
      </div>

      <h3 className="text-xl font-semibold text-slate-900 mb-2">Upload a PDF</h3>
      <p className="text-slate-500">Drop or click to select a patient record (.pdf)</p>

      <div className="mt-6 flex justify-center gap-4 text-xs text-slate-400 uppercase tracking-wider font-medium">
        <span className="flex items-center gap-1">
          <CheckCircle2 className="w-3 h-3" /> PDF only
        </span>
        <span className="flex items-center gap-1">
          <CheckCircle2 className="w-3 h-3" /> OCR enabled
        </span>
      </div>
    </div>
  );
};

const tabs = ['Overview', 'Patient', 'Problems', 'Meds', 'Labs', 'Procedures', 'Imaging', 'Notes', 'Everything'] as const;
type Tab = (typeof tabs)[number];

const SectionViewer = ({ activeTab, structuredData, loading }: { activeTab: Tab; structuredData: StructuredData | null; loading: boolean }) => {
  const formatPageRange = (pageRange?: number[]) => {
    if (!pageRange || pageRange.length === 0) return '—';
    if (pageRange.length === 1) return `Page ${pageRange[0]}`;
    return `Pages ${pageRange[0]}–${pageRange[pageRange.length - 1]}`;
  };

  const splitContent = (text: string) =>
    text
      .split(/\n+/)
      .map((line) => line.replace(/^[•\-\d\.\)\s]+/, '').trim())
      .filter(Boolean);

  const prettify = (str?: string) => {
    if (!str) return '';
    return str
      .replace(/_/g, ' ')
      .toLowerCase()
      .replace(/\b\w/g, (m) => m.toUpperCase());
  };

  const isStructuralType = (t: string) => {
    const lower = t.toLowerCase();
    return (
      lower.includes('header') ||
      lower.includes('section') ||
      lower.includes('page_metadata') ||
      lower.includes('demographics') ||
      lower.includes('margin_content') ||
      lower.includes('page_number')
    );
  };

  const isHeadingLine = (line: string) => {
    const l = line.trim();
    if (!l) return true;
    const lower = l.toLowerCase();
    const tokens = ['document summary', 'patient demographics', 'document date', 'key findings', 'medications', 'laboratory', 'procedures', 'other important information', 'problem list', 'past medical history'];
    if (tokens.some((t) => lower.startsWith(t))) return true;
    if (l.endsWith(':')) return true;
    if (l.length <= 3) return true;
    return false;
  };

  const cleanLines = (lines: string[]) => {
    const seen = new Set<string>();
    const filtered: string[] = [];
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (isHeadingLine(trimmed)) continue;
      const key = trimmed.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      filtered.push(trimmed);
    }
    return filtered;
  };

  const elementLabel = (type: string) => {
    const t = type.toLowerCase();
    if (t.includes('medication')) return 'Medication';
    if (t.includes('lab')) return 'Lab';
    if (t.includes('procedure')) return 'Procedure';
    if (t.includes('radiology')) return 'Imaging';
    if (t.includes('problem')) return 'Problem';
    if (t.includes('demographics')) return 'Demographics';
    if (t.includes('list')) return 'List';
    if (t.includes('header')) return 'Header';
    if (t.includes('form')) return 'Form';
    return 'Detail';
  };

  const elementPriority = (type: string) => {
    const t = type.toLowerCase();
    if (t.includes('header')) return 6;
    if (t.includes('problem')) return 5;
    if (t.includes('medication')) return 5;
    if (t.includes('lab')) return 5;
    if (t.includes('procedure')) return 5;
    if (t.includes('demographics')) return 4;
    if (t.includes('list')) return 3;
    if (t.includes('clinical')) return 3;
    if (t.includes('form')) return 2;
    return 1;
  };

  const lineMatchesTab = (tab: Tab, line: string, elementType: string) => {
    const lower = line.toLowerCase();
    const t = elementType.toLowerCase();
    const has = (keywords: string[]) => keywords.some((k) => lower.includes(k));
    const tHas = (keywords: string[]) => keywords.some((k) => t.includes(k));

    switch (tab) {
      case 'Meds':
        return tHas(['medication']) || has(['mg', 'mcg', 'units', 'dose', 'medication', 'tablet', 'pill', 'insulin', 'daily', 'q.']);
      case 'Labs':
        return tHas(['lab']) || has(['lab', 'result', 'test', 'panel', 'hemoglobin', 'hematocrit', 'cbc', 'creatinine', 'glucose']);
      case 'Procedures':
        return tHas(['procedure']) || has(['procedure', 'surgery', 'operative', 'biopsy', 'colectomy', 'performed']);
      case 'Imaging':
        return tHas(['radiology', 'imaging']) || has(['ct', 'mri', 'x-ray', 'radiology', 'ultrasound']);
      case 'Problems':
        return tHas(['problem', 'history']) || has(['history', 'diagnosis', 'problem', 'hx', 'assessment']);
      case 'Notes':
        return tHas(['progress', 'clinical']) || has(['note', 'plan', 'assessment']);
      default:
        return true;
    }
  };

  const extractLines = (section: SectionContent, tab: Tab, take = 8) => {
    const elements = section.pages?.flatMap((p) =>
      (p.elements || []).map((el) => ({ ...el, page_number: p.page_number || el.page_number }))
    ) || [];

    const ordered = [...elements].sort((a, b) => elementPriority(b.type) - elementPriority(a.type));
    const lines: { text: string; tag: string; page?: number }[] = [];

    for (const el of ordered) {
      if (isStructuralType(el.type || '')) continue;
      const filtered = cleanLines(splitContent(el.content || ''));
      filtered.forEach((line) => {
        if (lines.length >= take) return;
        if (!lineMatchesTab(tab, line, el.type || '')) return;
        lines.push({ text: line, tag: elementLabel(el.type), page: el.page_number });
      });
      if (lines.length >= take) break;
    }
    return lines;
  };

  const sectionMatches = (section: SectionContent, keywords: string[]) => {
    const type = (section.type || '').toLowerCase();
    const title = (section.title || '').toLowerCase();
    return keywords.some((k) => type.includes(k) || title.includes(k));
  };

  const sectionHasElementTypes = (section: SectionContent, types: string[]) => {
    const wanted = types.map((t) => t.toLowerCase());
    return (section.pages || []).some((p) => (p.elements || []).some((el) => wanted.includes((el.type || '').toLowerCase())));
  };

  const filterSections = (tab: Tab) => {
    const sections = structuredData?.sections || [];
    const matchers: Record<Tab, (s: SectionContent) => boolean> = {
      Overview: () => false,
      Patient: () => false,
      Problems: (s) =>
        sectionMatches(s, ['problem', 'history', 'diagnosis']) ||
        sectionHasElementTypes(s, ['problem_list', 'list_items']),
      Meds: (s) =>
        sectionMatches(s, ['medication', 'meds']) ||
        sectionHasElementTypes(s, ['medication_table']),
      Labs: (s) =>
        sectionMatches(s, ['laboratory', 'lab']) ||
        sectionHasElementTypes(s, ['lab_results_table']),
      Procedures: (s) =>
        sectionMatches(s, ['procedure', 'surgery', 'operative']) ||
        sectionHasElementTypes(s, ['general_table', 'procedure']),
      Imaging: (s) => sectionMatches(s, ['radiology', 'imaging', 'ct', 'mri', 'x-ray']),
      Notes: (s) => sectionMatches(s, ['progress', 'clinical_content', 'note', 'plan']),
      Everything: () => true,
    };
    const matcher = matchers[tab];
    return sections.filter(matcher);
  };

  const renderSectionGrid = (tab: Tab, sections: SectionContent[], emptyLabel: string) => {
    if (!sections || sections.length === 0) {
      return <p className="text-slate-500 italic text-sm">{emptyLabel}</p>;
    }
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sections.map((section) => {
          const lines = extractLines(section, tab);
          return (
            <div key={`${section.id || section.title}-${section.page_range || ''}`} className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold">{prettify(section.type) || 'Section'}</p>
                  <h4 className="text-sm font-semibold text-slate-900">{section.title || 'Untitled section'}</h4>
                  <p className="text-xs text-slate-500">{formatPageRange(section.page_range)}</p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  {section.page_count && (
                    <span className="text-[11px] text-blue-700 bg-blue-50 px-2 py-1 rounded-full font-semibold border border-blue-100">
                      {section.page_count}p
                    </span>
                  )}
                  {section.confidence && (
                    <span className="text-[11px] text-emerald-700 bg-emerald-50 px-2 py-1 rounded-full font-semibold border border-emerald-100">
                      {Math.round(section.confidence * 100)}% confident
                    </span>
                  )}
                </div>
              </div>
              <div className="mt-3 space-y-2">
                {lines.length > 0 ? (
                  lines.map((line, idx) => (
                    <div key={idx} className="flex items-start gap-2">
                      <span className="mt-1 w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                      <div className="flex-1">
                        <div className="text-xs uppercase tracking-wide text-slate-400 font-semibold flex items-center gap-2">
                          {line.tag}
                          {line.page && <span className="text-[10px] text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">Page {line.page}</span>}
                        </div>
                        <p className="text-sm text-slate-800 leading-snug">{line.text}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-500">No clinical content captured for this section.</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm h-full overflow-hidden flex flex-col">
        <div className="p-6 text-sm text-slate-600 flex-1 flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
        </div>
      </div>
    );
  }

  if (!structuredData) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm h-full overflow-hidden flex flex-col">
        <div className="p-6 text-sm text-slate-600 flex-1">
          <p className="mb-2 font-medium text-slate-700">No data yet.</p>
          <p className="text-slate-500">Upload or load a document to populate this section.</p>
        </div>
      </div>
    );
  }

  const summaryLines =
    structuredData.summary
      ?.split(/[\n•]+/)
      .map((s) => s.trim())
      .filter(Boolean) || [];

  const stats = [
    { label: 'Sections', value: structuredData.sections?.length ?? 0 },
    { label: 'Pages', value: structuredData.pages?.length ?? '—' },
    { label: 'Medications', value: structuredData.medications?.length ?? 0 },
    { label: 'Labs', value: structuredData.labs?.length ?? 0 },
    { label: 'Procedures', value: structuredData.procedures?.length ?? 0 },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col max-h-[65vh]">
      <div className="overflow-y-auto p-4 space-y-5">
        {activeTab === 'Overview' && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              {stats.map((s) => (
                <div key={s.label} className="rounded-xl border border-slate-200 bg-white px-3 py-3 shadow-sm">
                  <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold">{s.label}</p>
                  <p className="text-xl font-semibold text-slate-900">{s.value}</p>
                </div>
              ))}
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white text-slate-900 p-5 shadow-sm">
              <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Document summary</p>
              {cleanLines(summaryLines).length > 0 ? (
                <div className="space-y-2">
                  {cleanLines(summaryLines).map((line, idx) => (
                    <div key={idx} className="flex items-start gap-2">
                      <span className="mt-1 w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                      <p className="text-sm leading-relaxed">{line}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">No summary available.</p>
              )}
            </div>
            {structuredData.document_dates && structuredData.document_dates.length > 0 && (
              <div>
                <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Dates seen</p>
                <div className="flex flex-wrap gap-2">
                  {structuredData.document_dates.map((d) => (
                    <span key={d} className="px-2 py-1 rounded-lg border border-slate-200 bg-white text-xs text-slate-700 shadow-sm">
                      {d}
                    </span>
                  ))}
                </div>
              </div>
            )}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-slate-800">Document outline</h4>
                <p className="text-xs text-slate-500">Click into tabs to deep dive</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {(structuredData.sections || []).map((s) => (
                  <div key={s.id || s.title} className="px-3 py-2 rounded-full border border-slate-200 bg-white shadow-sm text-xs text-slate-700 flex items-center gap-2">
                    <span className="font-semibold text-slate-900">{s.title || 'Untitled'}</span>
                    <span className="text-[11px] text-slate-500">{formatPageRange(s.page_range)}</span>
                  </div>
                ))}
                {(structuredData.sections || []).length === 0 && <p className="text-xs text-slate-500">No outline available.</p>}
              </div>
            </div>
          </>
        )}

        {activeTab === 'Patient' && (
          <div className="space-y-4">
            <div className="rounded-xl border border-slate-200 p-4 bg-slate-50">
              <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Demographics</p>
              {structuredData.patient_demographics?.content ? (
                <div className="space-y-1">
                  {splitContent(structuredData.patient_demographics.content).map((line, idx) => (
                    <p key={idx} className="text-sm text-slate-800">{line}</p>
                  ))}
                  {structuredData.patient_demographics.page_number && (
                    <p className="text-[11px] text-slate-500 mt-2">Located on page {structuredData.patient_demographics.page_number}</p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-slate-500 italic">No patient demographic block captured.</p>
              )}
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Problem list / history</p>
              {renderSectionGrid(filterSections('Problems'), 'No diagnoses or problem lists detected.')}
            </div>
          </div>
        )}

        {activeTab === 'Problems' && (
          renderSectionGrid(filterSections('Problems'), 'No diagnoses or problem lists detected.')
        )}

        {activeTab === 'Meds' && (
          <div className="space-y-4">
            {structuredData.medications.length > 0 ? (
              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Medication</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Dosage</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Frequency</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Start</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">End</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Notes</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-slate-200">
                    {structuredData.medications.map((med, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="px-4 py-3 text-slate-900 font-semibold">{med.drug_name}</td>
                        <td className="px-4 py-3 text-slate-700">{med.dosage || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{med.frequency || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{med.start_date || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{med.end_date || '—'}</td>
                        <td className="px-4 py-3 text-slate-600 text-xs">{med.notes || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-slate-500 italic text-sm">No structured medications returned.</p>
            )}
            <div>
              <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Source sections</p>
              {renderSectionGrid('Meds', filterSections('Meds'), 'No medication sections detected in the document.')}
            </div>
          </div>
        )}

        {activeTab === 'Labs' && (
          <div className="space-y-4">
            {structuredData.labs.length > 0 ? (
              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Test</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Date</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Result</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Reason</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Notes</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-slate-200">
                    {structuredData.labs.map((lab, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="px-4 py-3 text-slate-900 font-semibold">{lab.test_name}</td>
                        <td className="px-4 py-3 text-slate-700">{lab.date_ordered || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{lab.result || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{lab.reason || '—'}</td>
                        <td className="px-4 py-3 text-slate-600 text-xs">{lab.notes || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-slate-500 italic text-sm">No structured labs returned.</p>
            )}
            <div>
              <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Source sections</p>
              {renderSectionGrid('Labs', filterSections('Labs'), 'No lab sections detected in the document.')}
            </div>
          </div>
        )}

        {activeTab === 'Procedures' && (
          <div className="space-y-4">
            {structuredData.procedures.length > 0 ? (
              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Procedure</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Date</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Purpose</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Result</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">Notes</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-slate-200">
                    {structuredData.procedures.map((proc, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="px-4 py-3 text-slate-900 font-semibold">{proc.procedure_name}</td>
                        <td className="px-4 py-3 text-slate-700">{proc.date || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{proc.purpose || '—'}</td>
                        <td className="px-4 py-3 text-slate-700">{proc.result || '—'}</td>
                        <td className="px-4 py-3 text-slate-600 text-xs">{proc.notes || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-slate-500 italic text-sm">No structured procedures returned.</p>
            )}
            <div>
              <p className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">Source sections</p>
              {renderSectionGrid('Procedures', filterSections('Procedures'), 'No procedure sections detected in the document.')}
            </div>
          </div>
        )}

        {activeTab === 'Imaging' && renderSectionGrid('Imaging', filterSections('Imaging'), 'No imaging or radiology sections detected.')}

        {activeTab === 'Notes' && renderSectionGrid('Notes', filterSections('Notes'), 'No clinical notes detected.')}

        {activeTab === 'Everything' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-500">Every grouped section from the ontology.</p>
            {renderSectionGrid('Everything', filterSections('Everything'), 'No sections captured.')}
          </div>
        )}
      </div>
    </div>
  );
};

const Modal = ({ open, onClose, children, title }: { open: boolean; onClose: () => void; children: React.ReactNode; title?: string }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 px-4">
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-800">{title || 'Details'}</h3>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-800">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
};

const AnalysisArea = ({
  docId,
  docMeta,
  processing,
  onQuery,
  recentQueries,
}: {
  docId: string;
  docMeta: DocumentResponse | null;
  processing: boolean;
  onQuery: (question: string) => Promise<QueryResponse>;
  recentQueries: RecentQuery[];
}) => {
  const [messages, setMessages] = useState<
    { role: 'user' | 'assistant'; content: string; reasoning?: string; evidence?: MatchedElement[] }[]
  >([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('Overview');
  const [chatOpen, setChatOpen] = useState(true);
  const [selectedEvidenceList, setSelectedEvidenceList] = useState<MatchedElement[] | null>(null);
  const [evidenceIndex, setEvidenceIndex] = useState(0);
  const [openReasoning, setOpenReasoning] = useState<Set<number>>(new Set());
  const [structuredData, setStructuredData] = useState<StructuredData | null>(null);
  const [loadingStructuredData, setLoadingStructuredData] = useState(false);

  const docReady = !!(docMeta && docMeta.extractions && docMeta.extractions.length > 0);

  useEffect(() => {
    if (docReady && messages.length === 0) {
      setMessages([
        {
          role: 'assistant',
          content: 'Extraction is ready. Ask a question about this document (e.g., medications, diagnoses, vitals).',
        },
      ]);
    }
  }, [docReady, docId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (docReady && docId) {
      setLoadingStructuredData(true);
      api.getStructuredData(docId)
        .then((data) => {
          setStructuredData(data);
        })
        .catch((err) => {
          console.error('Failed to fetch structured data:', err);
        })
        .finally(() => {
          setLoadingStructuredData(false);
        });
    }
  }, [docReady, docId]);

  const handleSend = async () => {
    if (!input.trim()) return;
    if (!docReady || !docId) {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Load a document with an extraction before querying.' }]);
      return;
    }
    const q = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setIsLoading(true);
    try {
      const res = await onQuery(q);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.answer_summary,
          reasoning: res.reasoning,
          evidence: res.matched_elements,
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: err?.message || 'Error querying the document. Ensure the backend is reachable.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReplay = (rq: RecentQuery) => {
    console.debug('Replaying cached query', rq.question);
    // If evidence lacks image_url, enrich from docMeta metadata
    const pageImages =
      docMeta?.extractions?.find((e) => e.metadata && e.metadata.page_images)?.metadata?.page_images || {};
    const pageDims =
      docMeta?.extractions?.find((e) => e.metadata && e.metadata.page_dimensions)?.metadata?.page_dimensions || {};

    const evidence =
      rq.evidence?.map((ev) => {
        const pageKey = ev.page ? String(ev.page) : ev.page_key ? String(ev.page_key) : '';
        const maybeImg = ev.image_url || (pageKey ? pageImages[pageKey] : undefined);
        const dims = pageKey ? pageDims[pageKey] : undefined;
        return {
          ...ev,
          image_url: maybeImg || ev.image_url,
          page_width_px: ev.page_width_px || dims?.width_px,
          page_height_px: ev.page_height_px || dims?.height_px,
        };
      }) || rq.evidence;

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: rq.question },
      {
        role: 'assistant',
        content: rq.answer,
        reasoning: rq.reasoning,
        evidence,
      },
    ]);
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-64px)] p-4 lg:p-6">
      <div className="flex-1 flex flex-col gap-6 overflow-y-auto">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">Extraction status</p>
              {docReady ? (
                <>
                  <p className="text-sm text-slate-800 mt-1">Ready for queries.</p>
                  <p className="text-xs text-slate-500 mt-1">
                    Artifacts: {docMeta?.extractions?.map((e) => e.kind).join(', ') || 'available'}
                  </p>
                </>
              ) : processing ? (
                <p className="text-sm text-slate-600 mt-1">Running pipeline…</p>
              ) : (
                <p className="text-sm text-slate-600 mt-1">Upload a PDF to start extraction.</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm">
          <div className="p-4 border-b border-slate-100 flex gap-2 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                  activeTab === tab
                    ? 'bg-blue-50 text-blue-700 border border-blue-100'
                    : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
          <div className="p-4">
            <SectionViewer activeTab={activeTab} structuredData={structuredData} loading={loadingStructuredData} />
            {docReady && docMeta?.document && (
              <div className="mt-3 text-xs text-slate-500 space-y-1">
                <div className="font-semibold text-slate-600">Document info</div>
                <div>ID: {docMeta.document.id}</div>
                {docMeta.document.original_filename && <div>File: {docMeta.document.original_filename}</div>}
                {docMeta.document.created_at && <div>Created: {docMeta.document.created_at}</div>}
              </div>
            )}
          </div>
        </div>
      </div>

      {chatOpen && (
        <div className="w-full lg:w-5/12 bg-white border border-slate-200 rounded-xl shadow-sm flex flex-col">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">Chat</p>
              <p className="text-sm text-slate-700">
                {docReady ? 'Ask about this record' : 'Upload or load a document to enable queries'}
              </p>
            </div>
            <button
              onClick={() => setChatOpen(false)}
              className="text-xs text-slate-500 hover:text-slate-800 border border-slate-200 px-3 py-1 rounded-lg"
            >
              Collapse
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth">
            {recentQueries.length > 0 && (
              <div className="text-xs text-slate-500 border border-slate-200 rounded-lg p-3 bg-slate-50">
                <div className="font-semibold text-slate-700 mb-2">Recent queries</div>
                <div className="space-y-1">
                  {recentQueries.map((rq, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleReplay(rq)}
                      className="w-full text-left text-slate-700 hover:text-blue-700 hover:bg-white rounded px-2 py-1 border border-transparent hover:border-blue-100"
                    >
                      {rq.question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.length === 0 && (
              <div className="text-sm text-slate-500 border border-dashed border-slate-200 rounded-lg p-4">
                No messages yet. Upload or load a document and ask a question to get started.
              </div>
            )}

            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`flex gap-4 animate-in slide-in-from-bottom-2 duration-300 ${
                  m.role === 'user' ? 'flex-row-reverse' : ''
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm ${
                    m.role === 'user' ? 'bg-slate-800 text-white' : 'bg-blue-100 text-blue-600'
                  }`}
                >
                  {m.role === 'user' ? <span className="text-xs font-bold">MD</span> : <Activity className="w-4 h-4" />}
                </div>

                <div className="max-w-[85%] space-y-3">
                  <div
                    className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                      m.role === 'user'
                        ? 'bg-slate-800 text-white rounded-tr-none'
                        : 'bg-white border border-slate-200 text-slate-800 rounded-tl-none'
                    }`}
                  >
                    {m.content}
                  </div>

                  {m.role === 'assistant' && (m.reasoning || (m.evidence && m.evidence.length > 0)) && (
                    <div className="flex items-center gap-3 text-xs text-slate-600">
                      {m.reasoning && (
                        <button
                          onClick={() => {
                            const next = new Set(openReasoning);
                            next.has(idx) ? next.delete(idx) : next.add(idx);
                            setOpenReasoning(next);
                          }}
                          className="px-2 py-1 rounded border border-slate-200 hover:border-blue-300 hover:text-blue-700 transition-colors"
                        >
                          {openReasoning.has(idx) ? 'Hide reasoning' : 'Show reasoning'}
                        </button>
                      )}
                      {m.evidence && m.evidence.length > 0 && (
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedEvidenceList(m.evidence || null);
                            setEvidenceIndex(0);
                          }}
                          className="inline-flex items-center gap-1 px-2 py-1 rounded border border-slate-200 hover:border-blue-300 hover:text-blue-700 transition-colors"
                        >
                          <BoxSelect className="w-3 h-3" />
                          View sources
                        </button>
                      )}
                    </div>
                  )}

                  {m.role === 'assistant' && m.reasoning && openReasoning.has(idx) && (
                    <div className="bg-slate-50 border border-slate-200 rounded-xl overflow-hidden text-xs">
                      <div className="px-4 py-2 bg-slate-100 border-b border-slate-200 flex items-center gap-2 text-slate-600 font-medium">
                        <BoxSelect className="w-3 h-3" />
                        Reasoning
                      </div>
                      <div className="p-4">
                        <p className="text-slate-600 leading-relaxed italic border-l-2 border-blue-200 pl-3">{m.reasoning}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-4">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Activity className="w-4 h-4 text-blue-600" />
                </div>
                <div className="bg-white p-4 rounded-xl rounded-tl-none border border-slate-200 shadow-sm flex items-center gap-3">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                  <span className="text-sm text-slate-500">Analyzing...</span>
                </div>
              </div>
            )}
            <div className="h-2" />
          </div>

          <div className="p-4 bg-white border-t border-slate-100">
            <div className="relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder={docReady ? 'Ask a question (e.g., medications?)' : 'Upload or load before querying'}
                disabled={isLoading || processing}
                className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder:text-slate-400 text-sm shadow-inner disabled:opacity-60"
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !input.trim() || processing || !docReady}
                className="absolute right-2 top-2 bottom-2 aspect-square bg-slate-900 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-[10px] text-slate-400 text-center mt-2">
              AI responses may be inaccurate. Verify with source documents.
            </p>
          </div>
        </div>
      )}

      {!chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 bg-slate-900 text-white px-4 py-2 rounded-full shadow-lg hover:bg-blue-600 transition-colors text-sm"
        >
          Open chat
        </button>
      )}

      <Modal open={!!selectedEvidenceList} onClose={() => setSelectedEvidenceList(null)} title="Provenance">
        {selectedEvidenceList && selectedEvidenceList.length > 0 ? (
          <div className="space-y-3 text-sm text-slate-700">
            <div className="flex items-center justify-between">
              <div className="font-semibold text-slate-800">Source {evidenceIndex + 1} of {selectedEvidenceList.length}</div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setEvidenceIndex((i) => Math.max(0, i - 1))}
                  disabled={evidenceIndex === 0}
                  className="px-2 py-1 text-xs border border-slate-200 rounded disabled:opacity-50"
                >
                  Prev
                </button>
                <button
                  onClick={() => setEvidenceIndex((i) => Math.min(selectedEvidenceList.length - 1, i + 1))}
                  disabled={evidenceIndex === selectedEvidenceList.length - 1}
                  className="px-2 py-1 text-xs border border-slate-200 rounded disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
            {(() => {
              const ev = selectedEvidenceList[evidenceIndex] || ({} as MatchedElement);
              const bbox = Array.isArray(ev.bbox) ? ev.bbox : [];
              return (
                <>
                  <div className="text-xs text-slate-500">
                    Page {ev.page ?? 'n/a'} • BBox: {bbox.length ? bbox.join(', ') : 'n/a'}
                  </div>
                  <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">{ev.text || 'No text available.'}</div>
                  {ev.image_url ? (
                    <div className="mt-2">
                      <div className="text-xs text-slate-500 mb-1">Preview</div>
                      <BBoxPreview ev={ev} />
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500">No page image available.</p>
                  )}
                </>
              );
            })()}
          </div>
        ) : null}
      </Modal>
    </div>
  );
};

export default function App() {
  const [docModalOpen, setDocModalOpen] = useState(false);
  const [docId, setDocId] = useState<string>('');
  const [docMeta, setDocMeta] = useState<DocumentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState<string | null>(null);
  const [recentDocs, setRecentDocs] = useState<RecentDoc[]>([]);
  const [recentQueries, setRecentQueries] = useState<RecentQuery[]>([]);

  useEffect(() => {
    setRecentDocs(loadLocal<RecentDoc[]>(RECENT_DOCS_KEY, []));
    setRecentQueries(loadLocal<RecentQuery[]>(RECENT_QUERIES_KEY, []));
  }, []);

  useEffect(() => saveLocal(RECENT_DOCS_KEY, recentDocs), [recentDocs]);
  useEffect(() => saveLocal(RECENT_QUERIES_KEY, recentQueries), [recentQueries]);

  const updateRecents = (meta: DocumentResponse) => {
    const next: RecentDoc = {
      id: meta.document.id,
      name: meta.document.original_filename,
      createdAt: meta.document.created_at,
    };
    setRecentDocs((prev) => {
      const filtered = prev.filter((d) => d.id !== next.id);
      return [next, ...filtered].slice(0, 10);
    });
  };

  const handleUpload = async (file: File) => {
    try {
      setDocModalOpen(false);
      setError(null);
      setProcessing(true);
      setProcessingStep('Uploading');
      setDocMeta(null);
      const uploadRes = await api.upload(file);
      setDocId(uploadRes.document_id);
      setProcessingStep('Extracting');
      await api.extract(uploadRes.document_id);
      setProcessingStep('Fetching extraction');
      const meta = await api.getDocument(uploadRes.document_id);
      setDocMeta(meta);
      updateRecents(meta);
      setDocModalOpen(false);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Error processing document.');
    } finally {
      setProcessing(false);
      setProcessingStep(null);
    }
  };

  const handleLoadExisting = async (id: string) => {
    try {
      setDocModalOpen(false);
      setError(null);
      setProcessing(true);
      setProcessingStep('Loading document');
      const meta = await api.getDocument(id);
      setDocId(id);
      setDocMeta(meta);
      updateRecents(meta);
      setDocModalOpen(false);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Unable to load document.');
    } finally {
      setProcessing(false);
      setProcessingStep(null);
    }
  };

  const handleQuery = async (question: string) => {
    if (!docId) throw new Error('No document loaded');
    const res = await api.query(docId, question);
    setRecentQueries((prev) => {
      const next: RecentQuery = {
        docId,
        question,
        answer: res.answer_summary,
        reasoning: res.reasoning,
        evidence: res.matched_elements,
      };
      return [next, ...prev.filter((q) => !(q.docId === docId && q.question === question))].slice(0, 20);
    });
    return res;
  };

  const docRecentQueries = recentQueries.filter((q) => q.docId === docId);

  return (
    <div className="min-h-screen bg-slate-100 font-sans text-slate-900 selection:bg-blue-200 selection:text-blue-900 flex flex-col">
      <Header />

      <div className="bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">Current document</p>
          <p className="text-sm text-slate-800">
            {docMeta?.document?.original_filename || docId || 'None selected'}
          </p>
          {processing && (
            <p className="text-xs text-blue-600 flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              {processingStep || 'Processing...'}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setDocModalOpen(true)}
            className="px-4 py-2 text-sm font-semibold rounded-lg bg-slate-900 text-white hover:bg-blue-600 transition-colors"
          >
            Select / Upload
          </button>
        </div>
      </div>

      <main className="flex-1 flex flex-col relative">
        <AnalysisArea
          docId={docId}
          docMeta={docMeta}
          processing={processing}
          onQuery={handleQuery}
          recentQueries={docRecentQueries}
        />

        {error && (
          <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-white border border-amber-300 text-amber-700 px-4 py-3 rounded-lg shadow-lg text-sm flex items-center gap-2">
            <Activity className="w-4 h-4" />
            <span>{error}</span>
            <button className="ml-3 text-xs text-slate-500 underline" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}
      </main>

      <Modal open={docModalOpen} onClose={() => setDocModalOpen(false)} title="Select or Upload Document">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <UploadCard onUpload={handleUpload} />
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">Recent documents</h3>
            <p className="text-xs text-slate-500 mb-3">Load without re-uploading.</p>
            <div className="space-y-2 max-h-[340px] overflow-y-auto">
              {recentDocs.length === 0 && <p className="text-sm text-slate-500">No recent documents yet.</p>}
              {recentDocs.map((d) => (
                <button
                  key={d.id}
                  onClick={() => handleLoadExisting(d.id)}
                  className="w-full text-left p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors relative"
                >
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setRecentDocs((prev) => prev.filter((x) => x.id !== d.id));
                    }}
                    className="absolute top-2 right-2 text-slate-400 hover:text-slate-700"
                    aria-label="Remove from recent"
                  >
                    <X className="w-3 h-3" />
                  </button>
                  <div className="text-sm font-semibold text-slate-800 truncate">{d.name || d.id}</div>
                  <div className="text-xs text-slate-500">{d.id}</div>
                  {d.createdAt && <div className="text-[11px] text-slate-400">Created: {d.createdAt}</div>}
                </button>
              ))}
            </div>
            <div className="mt-4">
              <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">Load by ID</p>
              <input
                type="text"
                placeholder="Document UUID"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const v = (e.target as HTMLInputElement).value.trim();
                    if (v) handleLoadExisting(v);
                  }
                }}
                className="mt-2 w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/40"
              />
            </div>
          </div>
        </div>
      </Modal>
    </div>
  );
}
// Helper to render bbox overlay on a loaded image
const BBoxPreview = ({ ev }: { ev: MatchedElement }) => {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const [natural, setNatural] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    setSize({ w: e.currentTarget.clientWidth, h: e.currentTarget.clientHeight });
    setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight });
  };
  const handleError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.warn('Image failed to load', ev.image_url, e);
  };

  // Keep image size in sync with viewport changes (bbox alignment depends on rendered size)
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const updateSize = () => setSize({ w: img.clientWidth, h: img.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(img);
    window.addEventListener('resize', updateSize);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, [ev.image_url]);

  // Use the source page dimensions from the API (DPI-correct), fallback to natural if missing.
  const baseW = ev.page_width_px || natural.w || 0;
  const baseH = ev.page_height_px || natural.h || 0;

  const bboxSource =
    (Array.isArray(ev.bbox) && ev.bbox.length === 4 && ev.bbox) ||
    (Array.isArray(ev.bbox_norm) && ev.bbox_norm.length === 4 && ev.bbox_norm) ||
    null;
  // Only treat as PDF-space when we're explicitly using bbox_norm (bottom-left origin).
  const bboxIsPdfSpace =
    !(Array.isArray(ev.bbox) && ev.bbox.length === 4) &&
    Array.isArray(ev.bbox_norm) &&
    ev.bbox_norm.length === 4;

  // Convert bbox into pixel coords using top-left origin (to match <img>)
  let overlayStyle: React.CSSProperties | null = null;
  if (bboxSource && size.w > 0 && size.h > 0 && baseW > 0 && baseH > 0) {
    const [r0, r1, r2, r3] = bboxSource.map((v) => Number(v) || 0);
    const normalized = bboxSource.every((v) => typeof v === "number" && v >= 0 && v <= 1);

    // If the detector reported boxes on a smaller canvas (e.g., model downscaled input),
    // use the per-page max bbox extent to upscale into the true PNG size.
    const coordMaxW = ev.page_bbox_max_x_px && ev.page_bbox_max_x_px > 0 ? ev.page_bbox_max_x_px : baseW;
    const coordMaxH = ev.page_bbox_max_y_px && ev.page_bbox_max_y_px > 0 ? ev.page_bbox_max_y_px : baseH;
    const coordScaleX = coordMaxW ? baseW / coordMaxW : 1;
    const coordScaleY = coordMaxH ? baseH / coordMaxH : 1;

    let x0 = r0;
    let y0 = r1;
    let x1 = r2;
    let y1 = r3;

    if (bboxIsPdfSpace && ev.page_width_pdf && ev.page_height_pdf) {
      // PDF coords are bottom-left; convert to pixels and flip Y
      const scaleX = baseW / ev.page_width_pdf;
      const scaleY = baseH / ev.page_height_pdf;
      x0 = r0 * scaleX;
      x1 = r2 * scaleX;
      y0 = (ev.page_height_pdf - r3) * scaleY;
      y1 = (ev.page_height_pdf - r1) * scaleY;
    } else if (normalized) {
      x0 = r0 * baseW;
      y0 = r1 * baseH;
      x1 = r2 * baseW;
      y1 = r3 * baseH;
    } else {
      // Raw pixel coords from VLM are top-left origin; only scale up from detector canvas.
      x0 = r0 * coordScaleX;
      x1 = r2 * coordScaleX;
      y0 = r1 * coordScaleY;
      y1 = r3 * coordScaleY;
    }

    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];

    // Aggressive expansion to favor recall (sections often extend beyond the header bbox)
    const currW = x1 - x0;
    const currH = y1 - y0;
    const targetW = Math.max(currW + baseW * 0.05, currW * 3.5, baseW * 0.18);
    const targetH = Math.max(currH + baseH * 0.12, currH * 4, baseH * 0.20);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    x0 = Math.max(0, cx - targetW / 2);
    x1 = Math.min(baseW, cx + targetW / 2);
    y0 = Math.max(0, cy - targetH / 2);
    y1 = Math.min(baseH, cy + targetH / 2);

    const scaleX = size.w / baseW;
    const scaleY = size.h / baseH;

    overlayStyle = {
      left: x0 * scaleX,
      top: y0 * scaleY,
      width: Math.max(1, (x1 - x0) * scaleX),
      height: Math.max(1, (y1 - y0) * scaleY),
    };
  }

  // Debug logger for alignment; check DevTools console while modal is open.
  useEffect(() => {
    if (!bboxSource) return;
    console.log('BBox debug', {
      page: ev.page,
      bboxSource,
      bboxIsPdfSpace,
      bbox_norm: ev.bbox_norm,
      page_width_px: ev.page_width_px,
      page_height_px: ev.page_height_px,
      page_width_pdf: ev.page_width_pdf,
      page_height_pdf: ev.page_height_pdf,
      natural,
      rendered: size,
      baseW,
      baseH,
      overlayStyle,
    });
  }, [bboxSource, bboxIsPdfSpace, ev, natural, size, baseW, baseH, overlayStyle]);

  return (
    <div className="inline-block rounded-md border border-slate-200 bg-slate-100 p-2">
      <div className="relative">
        <img
          ref={imgRef}
          src={ev.image_url}
          alt="Page preview"
          className="block max-h-[70vh] max-w-[90vw] rounded border border-slate-200 shadow-sm"
          onLoad={handleLoad}
          onError={handleError}
        />
        {overlayStyle && (
          <div className="pointer-events-none absolute inset-0">
            <div
              className="absolute border-2 border-red-500/80 bg-red-500/10 rounded-sm"
              style={overlayStyle}
            />
          </div>
        )}
      </div>
    </div>
  );
};
