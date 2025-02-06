/** @format */
"use client";

import Link from "next/link";
import React, { useEffect, useState } from "react";
import { BsArchiveFill } from "react-icons/bs";
import { FaEdit, FaChevronLeft } from "react-icons/fa";
import { FiEdit } from "react-icons/fi";
import { BsThreeDots } from "react-icons/bs";
import { TbMinusVertical } from "react-icons/tb";
import { cn } from "@/utils/cn";
import { usePathname, useRouter } from "next/navigation";

type Props = {};

type ChatSession = {
  id: string;
  title: string;
  timestamp: Date;
  messages: Message[];
};

type Message = {
  role: "user" | "assistant";
  content: string;
  references?: Array<{ source: string; page: number; text: string }>;
};

export default function Sidebar({}: Props) {
  const [isSidebar, setSidebar] = useState(true);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const router = useRouter();

  useEffect(() => {
    // Fetch or retrieve chat sessions from local storage or API
    const sessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
    setChatSessions(sessions);
  }, []);

  const handleNewChat = () => {
    // Generate a unique ID for the new chat session
    const newChatId = `chat-${Date.now()}`;
    const newChatSession: ChatSession = {
      id: newChatId,
      title: `Chat ${chatSessions.length + 1}`, // Default title
      timestamp: new Date(),
      messages: [], // Initialize an empty array for messages
    };

    // Update the chat sessions in state and local storage
    const updatedSessions = [newChatSession, ...chatSessions];
    setChatSessions(updatedSessions);
    localStorage.setItem("chatSessions", JSON.stringify(updatedSessions));

    // Redirect to the new chat session
    router.push(`/chat/${newChatId}`);
  };

  const deleteSession = (sessionId: string) => {
    const sessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
    const updatedSessions = sessions.filter((session: ChatSession) => session.id !== sessionId);
    setChatSessions(updatedSessions);
    localStorage.setItem("chatSessions", JSON.stringify(updatedSessions));
  };

  const renameSession = (sessionId: string, newTitle: string) => {
    const sessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
    const sessionToRename = sessions.find((session: ChatSession) => session.id === sessionId);

    if (sessionToRename) {
      sessionToRename.title = newTitle;
      setChatSessions([...sessions]);
      localStorage.setItem("chatSessions", JSON.stringify(sessions));
    }
  };

  const toggleSidebar = () => {
    setSidebar(!isSidebar);
  };

  const groupChatSessionsByDate = (sessions: ChatSession[]) => {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const lastWeek = new Date(today);
    lastWeek.setDate(lastWeek.getDate() - 7);

    return sessions.reduce((acc, session) => {
      const sessionDate = new Date(session.timestamp);
      let label = '';

      if (sessionDate.toDateString() === today.toDateString()) {
        label = 'Today';
      } else if (sessionDate.toDateString() === yesterday.toDateString()) {
        label = 'Yesterday';
      } else if (sessionDate > lastWeek) {
        label = 'Last Week';
      } else {
        label = 'Older';
      }

      if (!acc[label]) {
        acc[label] = [];
      }
      acc[label].push(session);
      return acc;
    }, {} as Record<string, ChatSession[]>);
  };

  const groupedSessions = groupChatSessionsByDate(chatSessions);

  return (
    <div
      className={cn("min-h-screen relative transition-all ", {
        "-translate-x-full": !isSidebar,
        "w-full max-w-[244px]": isSidebar
      })}
    >
      {isSidebar && (
        <div
          className={cn(
            "min-h-screen w-full pl-4 pr-6 pt-20 dark:bg-[#0D0D0D]"
          )}
        >
          {/* new chat btn */}
          <div className="absolute top-5 left-0 pl-4 pr-6 w-full">
            <button
              onClick={handleNewChat}
              className="flex dark:bg-[#0D0D0D] justify-between w-full items-center p-2 hover:bg-slate-800 rounded-lg transition-all"
            >
              <section className="flex items-center gap-2">
                {/* logo */}
                <div className="h-7 w-7 bg-white p-1 rounded-full">
                  <img src="/assets/chatgpt-log.svg" alt="" />
                </div>
                <p className="text-sm">New Chat</p>
              </section>
              <FiEdit className="text-white text-sm" />
            </button>
          </div>

          {/* timelines */}
          <div className="w-full flex flex-col gap-5">
            {Object.entries(groupedSessions).map(([label, sessions]) => (
              <Timeline
                key={label}
                label={label}
                sessions={sessions}
                deleteSession={deleteSession}
                renameSession={renameSession}
              />
            ))}
          </div>
        </div>
      )}
      <div className="absolute inset-y-0 right-[-30px] flex items-center justify-center w-[30px]">
        <button
          onClick={toggleSidebar}
          className="h-[100px] group text-gray-500 hover:text-white w-full flex items-center justify-center transition-all"
        >
          <FaChevronLeft className="hidden group-hover:flex text-xl delay-500 duration-500 ease-in-out transition-all" />
          <TbMinusVertical className="text-3xl group-hover:hidden delay-500 duration-500 ease-in-out transition-all" />
        </button>
      </div>
    </div>
  );
}

function Timeline({
  label,
  sessions,
  deleteSession,
  renameSession,
}: {
  label: string;
  sessions: ChatSession[];
  deleteSession: (sessionId: string) => void;
  renameSession: (sessionId: string, newTitle: string) => void;
}) {
  const pathName = usePathname();
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");

  const handleRename = (sessionId: string) => {
    if (newTitle.trim()) {
      renameSession(sessionId, newTitle);
      setEditingSessionId(null);
      setNewTitle("");
    }
  };

  return (
    <div className="w-full flex flex-col">
      <p className="text-sm text-gray-500 font-bold p-2">{label}</p>
      {sessions.map((session) => (
        <div key={session.id} className="group relative">
          <Link
            href={`/chat/${session.id}`}
            className={cn(
              "p-2 ease-in-out duration-300 hover:bg-slate-800 rounded-lg transition-all items-center text-sm w-full flex justify-between",
              { "bg-slate-800": `/chat/${session.id}` === pathName }
            )}
          >
            <div className="text-ellipsis overflow-hidden w-[80%] whitespace-nowrap">
              {editingSessionId === session.id ? (
                <input
                  type="text"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="bg-transparent outline-none"
                  autoFocus
                />
              ) : (
                session.title
              )}
            </div>
            <div className="transition-all items-center gap-2 hidden group-hover:flex ease-in-out duration-300">
              <BsThreeDots />
              <BsArchiveFill />
            </div>
          </Link>
          <div className="absolute right-2 top-2 hidden group-hover:flex gap-2">
            <button
              onClick={() => deleteSession(session.id)}
              className="text-red-500 hover:text-red-700"
            >
              Delete
            </button>
            <button
              onClick={() => {
                setEditingSessionId(session.id);
                setNewTitle(session.title);
              }}
              className="text-blue-500 hover:text-blue-700"
            >
              Rename
            </button>
            {editingSessionId === session.id && (
              <button
                onClick={() => handleRename(session.id)}
                className="text-green-500 hover:text-green-700"
              >
                Save
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}