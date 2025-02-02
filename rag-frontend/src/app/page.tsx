'use client';
import { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [query, setQuery] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/query', { query });
      setConversation((prev) => [...prev, { query, response: res.data }]);
    } catch (error) {
      console.error('Error fetching data:', error);
      setConversation((prev) => [
        ...prev,
        { query, response: { error: 'An error occurred while processing your query.' } },
      ]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-6 text-center">RAG Inference Interface</h1>
      <form onSubmit={handleSubmit} className="mb-6">
        <label htmlFor="query" className="block text-lg font-medium mb-2">
          Enter your query:
        </label>
        <input
          id="query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything..."
          className="w-full p-3 border border-gray-700 rounded-md focus:outline-none focus:border-blue-500 bg-transparent text-white"
        />
        <button
          type="submit"
          disabled={loading}
          className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400"
        >
          {loading ? 'Loading...' : 'Submit'}
        </button>
      </form>

      {conversation.length > 0 && (
        <div className="mt-8">
          {conversation.map((item, index) => (
            <div key={index} className="mb-8">
              {/* Query */}
              <div className="mb-2">
                <span className="font-medium text-blue-400">Q:</span>{' '}
                <span className="text-gray-300">{item.query}</span>
              </div>

              {/* Response */}
              <div className="answer">
                <span className="font-medium text-green-400">A:</span>{' '}
                {item.response.error ? (
                  <div className="text-red-400">{item.response.error}</div>
                ) : (
                  <>
                    <p className="text-gray-300">{item.response.answer}</p>

                    {/* Citations */}
                    {item.response.references && item.response.references.length > 0 && (
                      <div className="references mt-4">
                        <h3 className="text-lg font-medium mb-2">References:</h3>
                        <ul className="list-disc pl-6">
                          {item.response.references.map((ref, idx) => (
                            <li key={idx} className="text-gray-400">
                              <a
                                href={ref.source}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-400 hover:underline"
                              >
                                {ref.source.split('/').pop()}
                              </a>
                              <br />
                              <span className="text-sm">
                                Page: {ref.page}, Text: {ref.text.slice(0, 100)}...
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Confidence Score */}
                    {item.response.confidence_score !== undefined && (
                      <div className="confidence-score mt-2">
                        <h3 className="text-lg font-medium">Confidence Score:</h3>
                        <p className="text-gray-300">{item.response.confidence_score.toFixed(2)}</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}