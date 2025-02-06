/** @format */
"use client"; // Add this directive at the top

import { FaChevronDown, FaArrowUp } from "react-icons/fa";
import { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  references?: Array<{ source: string; page: number; text: string }>;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      // Add user's message to the chat
      setMessages((prev) => [...prev, { role: "user", content: input }]);

      // Send the query to the backend
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response from the server.");
      }

      const data = await response.json();

      // Add the bot's response to the chat
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          references: data.references,
        },
      ]);
    } catch (err) {
      setError("An error occurred while processing your request.");
      console.error(err);
    } finally {
      setIsLoading(false);
      setInput("");
    }
  };

  return (
    <div className="h-full flex flex-col justify-between gap-3 pb-5">
      {/* nav */}
      <button className="text-lg font-bold flex items-center gap-2 rounded-xl p-2 hover:bg-slate-800 transition-all w-fit">
        <p>Select your own model</p>
        <FaChevronDown className="text-xs text-gray-500" />
      </button>

      {/* main */}
      <main className="flex flex-col gap-4 overflow-y-auto flex-grow">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex flex-col ${
              message.role === "user" ? "items-end" : "items-start"
            }`}
          >
            <div
              className={`max-w-2xl p-4 rounded-lg ${
                message.role === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-700 text-white"
              }`}
            >
              <p>{message.content}</p>
              {message.references && (
                <div className="mt-2">
                  <h4 className="font-bold">References:</h4>
                  <ul className="list-disc pl-5">
                    {message.references.map((ref, idx) => (
                      <li key={idx}>
                        <a
                          href={ref.source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-300 hover:underline"
                        >
                          {ref.source} (Page {ref.page})
                        </a>
                        <p className="text-sm text-gray-300">{ref.text}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          </div>
        )}
        {error && (
          <div className="text-red-500 text-center">
            <p>{error}</p>
          </div>
        )}
      </main>

      {/* bottom section */}
      <section className="max-w-3xl mx-auto flex flex-col gap-5">
        {/* Searchbar */}
        <div className="flex relative">
          <input
            type="text"
            placeholder="Ask anything..."
            className="w-[1000px] h-20 bg-inherit rounded-xl border border-gray-500 px-4 text-lg"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
          />
          <button
            className="text-black hover:opacity-80 bg-slate-500 w-fit rounded-xl p-3 absolute right-2 top-1"
            onClick={handleSendMessage}
            disabled={isLoading}
          >
            <FaArrowUp />
          </button>
        </div>
      </section>
    </div>
  );
}