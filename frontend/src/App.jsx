import ChatInterface from './components/ChatInterface';
import UploadPanel from './components/UploadPanel';
import './index.css';

export default function App() {
    return (
        <div className="app-layout">
            {/* Sidebar — Document Upload */}
            <aside className="sidebar">
                <UploadPanel />
            </aside>

            {/* Main — Chat Interface */}
            <main className="main-content">
                <ChatInterface />
            </main>
        </div>
    );
}
