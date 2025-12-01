import React, { useRef, useState } from 'react';
import { Upload, Send, Activity, Loader2, CheckCircle2, BoxSelect } from 'lucide-react';

/**
 * API base: set VITE_API_URL in your env (.env.local) for dev.
 * Fallback is the current Cloud Run host.
 */
const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ||
  'https://pdf2ehr-api3-3bf3r3croq-uw.a.run.app';

type MatchedElement = {
  page: number;
  bbox: number[];
  text: string;
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
      <span className="text-slate-400 truncate max-w-[200px]">{API_BASE}</span>
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
        <p className="text-slate-500">Upload a PDF and run extraction to view structured content here.</p>
      </div>
    </div>
  );
};

const DashboardView = ({
  docId,
  onUpload,
  processing,
}: {
  docId: string;
  onUpload: (f: File) => void;
  processing: boolean;
}) => {
  const [messages, setMessages] = useState<
    { role: 'user' | 'assistant'; content: string; reasoning?: string; evidence?: MatchedElement[] }[]
  >([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('Summary');
  const [chatOpen, setChatOpen] = useState(true);

  const handleSend = async () => {
    if (!input.trim()) return;
    if (!docId) {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Upload and extract a document before querying.' }]);
      return;
    }
    const q = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setIsLoading(true);

    try {
      const res = await api.query(docId, q);
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

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-64px)] p-4 lg:p-6">
      <div className="flex-1 flex flex-col gap-6 overflow-y-auto">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">Document</p>
            <p className="text-sm text-slate-700">{docId ? `ID: ${docId}` : 'No document uploaded yet'}</p>
          </div>
          {processing && (
            <div className="flex items-center gap-2 text-blue-600 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing...</span>
            </div>
          )}
        </div>

        <UploadCard onUpload={onUpload} />

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
          </div>
        </div>
      </div>

      {chatOpen && (
        <div className="w-full lg:w-5/12 bg-white border border-slate-200 rounded-xl shadow-sm flex flex-col">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold">Chat</p>
              <p className="text-sm text-slate-700">
                {docId ? 'Ask about this record' : 'Upload a document to enable queries'}
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
            {messages.length === 0 && (
              <div className="text-sm text-slate-500 border border-dashed border-slate-200 rounded-lg p-4">
                No messages yet. Upload a PDF and ask a question to get started.
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
                              <div
                                key={i}
                                className="group flex gap-3 items-start bg-white p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:shadow-md transition-all cursor-pointer"
                              >
                                <div className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-[10px] font-mono font-bold whitespace-nowrap group-hover:bg-blue-600 group-hover:text-white transition-colors">
                                  Pg {ev.page}
                                </div>
                                <div className="text-slate-700 group-hover:text-slate-900 font-medium">"{ev.text}"</div>
                              </div>
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
                placeholder={docId ? 'Ask a question (e.g., medications?)' : 'Upload and extract before querying'}
                disabled={isLoading || processing}
                className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder:text-slate-400 text-sm shadow-inner disabled:opacity-60"
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !input.trim() || processing || !docId}
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
    </div>
  );
};

export default function App() {
  const [docId, setDocId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);

  const handleUpload = async (file: File) => {
    try {
      setError(null);
      setProcessing(true);
      const uploadRes = await api.upload(file);
      setDocId(uploadRes.document_id);
      await api.extract(uploadRes.document_id);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Error processing document.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 font-sans text-slate-900 selection:bg-blue-200 selection:text-blue-900 flex flex-col">
      <Header />

      <main className="flex-1 flex flex-col relative">
        <DashboardView docId={docId} onUpload={handleUpload} processing={processing} />

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
