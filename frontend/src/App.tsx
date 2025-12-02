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
  image_url?: string; // optional future hook for page images
};

type QueryResponse = {
  question: string;
  answer_summary: string;
  reasoning: string;
  matched_elements: MatchedElement[];
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
};

const Header = () => (
  <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-md">
    <div className="flex items-center gap-3">
      <div>
        <h1 className="text-lg font-semibold text-slate-100 tracking-tight leading-none">EHRX</h1>
        <p className="text-xs text-slate-400 mt-0.5">PDF → structured EHR extraction</p>
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

const tabs = ['Summary', 'Meds', 'Labs', 'Procedures'] as const;
type Tab = (typeof tabs)[number];

const SectionViewer = ({ activeTab }: { activeTab: Tab }) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm h-full overflow-hidden flex flex-col">
      <div className="p-4 border-b border-slate-100 bg-slate-50">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Section</p>
        <p className="text-sm text-slate-700 mt-1">{activeTab}</p>
      </div>
      <div className="p-6 text-sm text-slate-600 flex-1">
        <p className="mb-2 font-medium text-slate-700">No data yet.</p>
        <p className="text-slate-500">Upload or load a document to populate this section.</p>
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

const DocumentsTab = ({
  onUpload,
  recentDocs,
  onLoadExisting,
  processing,
}: {
  onUpload: (f: File) => void;
  recentDocs: RecentDoc[];
  onLoadExisting: (id: string) => void;
  processing: boolean;
}) => {
  const [existingId, setExistingId] = useState('');

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      <div className="flex-1">
        <UploadCard onUpload={onUpload} />
      </div>
      <div className="w-full lg:w-1/3 bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        <h3 className="text-sm font-semibold text-slate-800 mb-2">Recent documents</h3>
        <p className="text-xs text-slate-500 mb-3">Load without re-uploading.</p>
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {recentDocs.length === 0 && <p className="text-sm text-slate-500">No recent documents yet.</p>}
          {recentDocs.map((d) => (
            <button
              key={d.id}
              onClick={() => onLoadExisting(d.id)}
              className="w-full text-left p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
            >
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
            value={existingId}
            onChange={(e) => setExistingId(e.target.value)}
            placeholder="Document UUID"
            className="mt-2 w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/40"
          />
          <button
            onClick={() => existingId && onLoadExisting(existingId.trim())}
            disabled={!existingId || processing}
            className="mt-2 w-full inline-flex items-center justify-center px-4 py-2 text-sm font-medium rounded-lg bg-slate-900 text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Load
          </button>
        </div>
      </div>
    </div>
  );
};

const AnalysisTab = ({
  docId,
  docMeta,
  processing,
  onQuery,
  recentQueries,
  onReplayQuery,
}: {
  docId: string;
  docMeta: DocumentResponse | null;
  processing: boolean;
  onQuery: (question: string) => Promise<void>;
  recentQueries: RecentQuery[];
  onReplayQuery: (rq: RecentQuery) => void;
}) => {
  const [messages, setMessages] = useState<
    { role: 'user' | 'assistant'; content: string; reasoning?: string; evidence?: MatchedElement[] }[]
  >([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('Summary');
  const [chatOpen, setChatOpen] = useState(true);
  const [selectedEvidence, setSelectedEvidence] = useState<MatchedElement | null>(null);

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
    await onQuery(q)
      .then((res: any) => {
        // onQuery already handled API; result is attached via event below (set externally)
      })
      .catch((err: any) => {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: err?.message || 'Error querying the document. Ensure the backend is reachable.',
          },
        ]);
      })
      .finally(() => setIsLoading(false));
  };

  const handleReplay = (rq: RecentQuery) => {
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: rq.question },
      {
        role: 'assistant',
        content: rq.answer,
        reasoning: rq.reasoning,
        evidence: rq.evidence,
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
            <SectionViewer activeTab={activeTab} />
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

                  {m.reasoning && (
                    <div className="bg-slate-50 border border-slate-200 rounded-xl overflow-hidden text-xs">
                      <div className="px-4 py-2 bg-slate-100 border-b border-slate-200 flex items-center gap-2 text-slate-600 font-medium">
                        <BoxSelect className="w-3 h-3" />
                        Reasoning
                      </div>
                      <div className="p-4">
                        <p className="text-slate-600 mb-4 leading-relaxed italic border-l-2 border-blue-200 pl-3">{m.reasoning}</p>

                        {m.evidence && m.evidence.length > 0 && (
                          <div className="space-y-2">
                            <p className="font-bold text-slate-400 uppercase tracking-wider text-[10px] mb-2">Source</p>
                            {m.evidence.map((ev, i) => (
                              <button
                                key={i}
                                onClick={() => setSelectedEvidence(ev)}
                                className="w-full text-left group flex gap-3 items-start bg-white p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:shadow-md transition-all cursor-pointer"
                              >
                                <div className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-[10px] font-mono font-bold whitespace-nowrap group-hover:bg-blue-600 group-hover:text-white transition-colors">
                                  Pg {ev.page}
                                </div>
                                <div className="text-slate-700 group-hover:text-slate-900 font-medium">"{ev.text}"</div>
                              </button>
                            ))}
                          </div>
                        )}
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

      <Modal open={!!selectedEvidence} onClose={() => setSelectedEvidence(null)} title="Provenance">
        {selectedEvidence ? (
          <div className="space-y-3 text-sm text-slate-700">
            <div className="font-semibold text-slate-800">Page {selectedEvidence.page}</div>
            <div className="text-xs text-slate-500">BBox: {selectedEvidence.bbox.join(', ')}</div>
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">{selectedEvidence.text}</div>
            {selectedEvidence.image_url && (
              <div className="mt-2">
                <div className="text-xs text-slate-500 mb-1">Preview</div>
                <div className="relative">
                  <img src={selectedEvidence.image_url} alt="Page preview" className="max-h-96 rounded border border-slate-200" />
                  {/* In future, overlay bbox here */}
                </div>
              </div>
            )}
            {!selectedEvidence.image_url && <p className="text-xs text-slate-500">No page image available.</p>}
          </div>
        ) : null}
      </Modal>
    </div>
  );
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'documents' | 'analysis'>('documents');
  const [docId, setDocId] = useState<string>('');
  const [docMeta, setDocMeta] = useState<DocumentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
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
      setError(null);
      setProcessing(true);
      setDocMeta(null);
      const uploadRes = await api.upload(file);
      setDocId(uploadRes.document_id);
      await api.extract(uploadRes.document_id);
      const meta = await api.getDocument(uploadRes.document_id);
      setDocMeta(meta);
      updateRecents(meta);
      setActiveTab('analysis');
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Error processing document.');
    } finally {
      setProcessing(false);
    }
  };

  const handleLoadExisting = async (id: string) => {
    try {
      setError(null);
      setProcessing(true);
      const meta = await api.getDocument(id);
      setDocId(id);
      setDocMeta(meta);
      updateRecents(meta);
      setActiveTab('analysis');
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Unable to load document.');
    } finally {
      setProcessing(false);
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
    // Append to messages via local effect: handled in AnalysisTab by replaying recent queries, or by returning result here
    return res;
  };

  const docRecentQueries = recentQueries.filter((q) => q.docId === docId);

  return (
    <div className="min-h-screen bg-slate-100 font-sans text-slate-900 selection:bg-blue-200 selection:text-blue-900 flex flex-col">
      <Header />

      <div className="border-b border-slate-200 bg-white px-4">
        <div className="flex gap-4">
          <button
            onClick={() => setActiveTab('documents')}
            className={`py-3 text-sm font-semibold border-b-2 ${
              activeTab === 'documents' ? 'border-blue-600 text-blue-700' : 'border-transparent text-slate-500'
            }`}
          >
            Documents
          </button>
          <button
            onClick={() => setActiveTab('analysis')}
            className={`py-3 text-sm font-semibold border-b-2 ${
              activeTab === 'analysis' ? 'border-blue-600 text-blue-700' : 'border-transparent text-slate-500'
            }`}
          >
            Analysis
          </button>
        </div>
      </div>

      <main className="flex-1 flex flex-col relative">
        {activeTab === 'documents' && (
          <div className="p-4 lg:p-6">
            <DocumentsTab onUpload={handleUpload} recentDocs={recentDocs} onLoadExisting={handleLoadExisting} processing={processing} />
          </div>
        )}
        {activeTab === 'analysis' && (
          <AnalysisTab
            docId={docId}
            docMeta={docMeta}
            processing={processing}
            onQuery={handleQuery}
            recentQueries={docRecentQueries}
            onReplayQuery={(rq) => {
              // handled inside AnalysisTab via handleReplay
            }}
          />
        )}

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
    </div>
  );
}
