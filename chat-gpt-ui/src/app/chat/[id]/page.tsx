/** @format */
"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { FaArrowUp } from "react-icons/fa";

type Message = {
  role: "user" | "assistant";
  content: string;
  references?: Array<{ source: string; page: number; text: string }>;
};

type ChatSession = {
  id: string;
  title: string;
  timestamp: Date;
  messages: Message[];
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { id } = useParams(); // Get the chat session ID from the URL
  const router = useRouter();

  useEffect(() => {
    console.log("Loading session with ID:", id); // Debugging
    const sessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
    console.log("All sessions:", sessions); // Debugging

    const currentSession = sessions.find((session: ChatSession) => session.id === id);
    console.log("Current session:", currentSession); // Debugging

    if (currentSession) {
      console.log("Loaded messages:", currentSession.messages); // Debugging
      setMessages(currentSession.messages);
    } else {
      console.log("Session not found, redirecting..."); // Debugging
      router.push("/");
    }
  }, [id, router]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      // Add user's message to the chat
      const newMessage: Message = { role: "user", content: input };
      setMessages((prevMessages) => [...prevMessages, newMessage]);

      // Save the message to the chat session
      const sessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
      const currentSession = sessions.find((session: ChatSession) => session.id === id);

      if (currentSession) {
        // If this is the first message, set it as the chat title
        if (currentSession.messages.length === 0) {
          currentSession.title = input.substring(0, 50); // Truncate to 50 characters
        }

        currentSession.messages = [...currentSession.messages, newMessage];
        localStorage.setItem("chatSessions", JSON.stringify(sessions));
        console.log("Updated sessions:", sessions); // Debugging
      }

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
      const botMessage: Message = {
        role: "assistant",
        content: data.answer,
        references: data.references,
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

      // Save the bot's response to the chat session
      if (currentSession) {
        currentSession.messages = [...currentSession.messages, botMessage];
        localStorage.setItem("chatSessions", JSON.stringify(sessions));
        console.log("Final sessions:", sessions); // Debugging
      }
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
      {/* Chat interface */}
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

      {/* Searchbar */}
      <section className="max-w-3xl mx-auto flex flex-col gap-5">
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