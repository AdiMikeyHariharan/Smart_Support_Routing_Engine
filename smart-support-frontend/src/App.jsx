import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [subject, setSubject] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);

      const response = await axios.post("/api/ticket_m2", {
        subject,
        description,
      });

      alert(
        `Ticket submitted!\nTicket ID: ${response.data.ticket_id}`
      );

      setSubject("");
      setDescription("");

    } catch (error) {
      console.error(error);
      alert("Failed to send ticket. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1 className="title">Smart Support</h1>
        <p className="subtitle">
          Create a support token to address your issues
        </p>

        <form onSubmit={handleSubmit} className="form">
          <div className="field">
            <label className="label">Token Subject</label>
            <input
              type="text"
              placeholder="Enter issue subject"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              className="input"
              required
            />
          </div>

          <div className="field">
            <label className="label">Token Description</label>
            <textarea
              placeholder="Describe the issue in detail..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="textarea"
              rows={4}
              required
            />
          </div>

          <button type="submit" className="button" disabled={loading}>
            {loading ? "Sending..." : "Submit Token"}
          </button>
        </form>
      </div>
    </div>
  );
}