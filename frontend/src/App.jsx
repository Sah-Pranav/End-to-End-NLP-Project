import { useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [text, setText] = useState('')
  const [summary, setSummary] = useState('')
  const [modelName, setModelName] = useState('')
  const [latencyMs, setLatencyMs] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSummarize() {
    const inputText = text.trim()

    if (!inputText) {
      return
    }

    setIsLoading(true)
    setError('')
    setSummary('')
    setModelName('')
    setLatencyMs(null)

    try {
      const response = await fetch(
        `${API_URL}/summarize`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: inputText,
          }),
        },
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ||
            'Unable to generate the summary.',
        )
      }

      setSummary(data.summary)
      setModelName(data.model_name)
      setLatencyMs(data.latency_ms)
    } catch (error) {
      if (error instanceof TypeError) {
        setError(
          'Unable to connect to the API. ' +
            'Please make sure the backend server is running.',
        )
      } else {
        setError(
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred.',
        )
      }
    } finally {
      setIsLoading(false)
    }
  }

  function handleClear() {
    setText('')
    setSummary('')
    setModelName('')
    setLatencyMs(null)
    setError('')
  }

  return (
    <main className="app">
      <header className="header">
        <p className="eyebrow">
          NLP PROJECT
        </p>

        <h1>Text Summarization</h1>

        <p className="subtitle">
          Generate concise summaries using a
          fine-tuned language model.
        </p>
      </header>

      <section
        className="summarization-card"
        aria-label="Text summarization"
      >
        <div className="input-section">
          <div className="section-header">
            <label htmlFor="input-text">
              Input text
            </label>

            <span className="character-count">
              {text.length} characters
            </span>
          </div>

          <textarea
            id="input-text"
            value={text}
            onChange={(event) => {
              setText(event.target.value)
            }}
            placeholder={
              'Paste or write the text you want to ' +
              'summarize...'
            }
            rows="12"
            disabled={isLoading}
          />

          <div className="input-footer">
            <button
              type="button"
              className="secondary-button"
              onClick={handleClear}
              disabled={
                isLoading ||
                (!text &&
                  !summary &&
                  !error)
              }
            >
              Clear
            </button>

            <button
              type="button"
              className="primary-button"
              onClick={handleSummarize}
              disabled={
                !text.trim() || isLoading
              }
            >
              {isLoading ? (
                <>
                  <span
                    className="spinner"
                    aria-hidden="true"
                  />
                  Summarizing...
                </>
              ) : (
                'Summarize'
              )}
            </button>
          </div>
        </div>

        <div className="output-section">
          <div className="section-header">
            <label htmlFor="summary">
              Generated summary
            </label>
          </div>

          <div
            id="summary"
            className="summary-box"
            aria-live="polite"
          >
            {isLoading && (
              <div className="loading-state">
                <span
                  className="spinner large-spinner"
                  aria-hidden="true"
                />
                <p>
                  Generating your summary...
                </p>
              </div>
            )}

            {!isLoading && error && (
              <div
                className="error-message"
                role="alert"
              >
                <strong>
                  Unable to generate summary
                </strong>

                <p>{error}</p>
              </div>
            )}

            {!isLoading &&
              !error &&
              !summary && (
                <p className="placeholder">
                  Your generated summary will
                  appear here.
                </p>
              )}

            {!isLoading &&
              summary && (
                <p className="summary-text">
                  {summary}
                </p>
              )}
          </div>

          {summary && (
            <div className="metadata">
              <div className="metadata-item">
                <span className="metadata-label">
                  Model
                </span>

                <span>
                  {modelName}
                </span>
              </div>

              {latencyMs !== null && (
                <div className="metadata-item">
                  <span className="metadata-label">
                    Latency
                  </span>

                  <span>
                    {latencyMs.toFixed(2)} ms
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </main>
  )
}

export default App