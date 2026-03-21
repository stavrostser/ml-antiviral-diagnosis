import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type CategoryField =
  | 'PATIENT_GENDER'
  | 'PHYSICIAN_TYPE'
  | 'PHYSICIAN_STATE'
  | 'LOCATION_TYPE'
  | 'INSURANCE_TYPE'
  | 'CONTRAINDICATIONS'
  | 'UNDERLYING_CONDITIONS'

type CategoryOptions = Record<CategoryField, string[]>

type FormState = {
  PATIENT_AGE: string
  PATIENT_GENDER: string
  NUM_CONDITIONS: string
  PHYSICIAN_TYPE: string
  PHYSICIAN_STATE: string
  LOCATION_TYPE: string
  INSURANCE_TYPE: string
  CONTRAINDICATIONS: string
  UNDERLYING_CONDITIONS: string[]
}

type PredictionResponse = {
  high_risk: boolean
  message: string
  prediction: number | null
  predicted_probability: number | null
  threshold: number | null
  model_filename: string | null
}

const categoryFields: Array<{ key: CategoryField; label: string }> = [
  { key: 'PATIENT_GENDER', label: 'Patient gender' },
  { key: 'PHYSICIAN_TYPE', label: 'Physician type' },
  { key: 'PHYSICIAN_STATE', label: 'Physician state' },
  { key: 'LOCATION_TYPE', label: 'Location type' },
  { key: 'INSURANCE_TYPE', label: 'Insurance type' },
  { key: 'CONTRAINDICATIONS', label: 'Contraindications' },
]

const emptyForm: FormState = {
  PATIENT_AGE: '71',
  PATIENT_GENDER: '',
  NUM_CONDITIONS: '0',
  PHYSICIAN_TYPE: '',
  PHYSICIAN_STATE: '',
  LOCATION_TYPE: '',
  INSURANCE_TYPE: '',
  CONTRAINDICATIONS: '',
  UNDERLYING_CONDITIONS: [],
}

function chooseDefault(options: string[], preferredValues: string[]): string {
  for (const preferredValue of preferredValues) {
    if (options.includes(preferredValue)) {
      return preferredValue
    }
  }

  return options[0] ?? ''
}

function createDefaultForm(options: CategoryOptions): FormState {
  return {
    PATIENT_AGE: '71',
    PATIENT_GENDER: chooseDefault(options.PATIENT_GENDER, ['M', 'F']),
    NUM_CONDITIONS: '0',
    PHYSICIAN_TYPE: chooseDefault(options.PHYSICIAN_TYPE, ['UNSPECIFIED']),
    PHYSICIAN_STATE: chooseDefault(options.PHYSICIAN_STATE, ['UNSPECIFIED']),
    LOCATION_TYPE: chooseDefault(options.LOCATION_TYPE, ['INDEPENDENT LABORATORY', 'OFFICE']),
    INSURANCE_TYPE: chooseDefault(options.INSURANCE_TYPE, ['COMMERCIAL']),
    CONTRAINDICATIONS: chooseDefault(options.CONTRAINDICATIONS, ['Unspecified']),
    UNDERLYING_CONDITIONS: [],
  }
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function App() {
  const [options, setOptions] = useState<CategoryOptions | null>(null)
  const [formState, setFormState] = useState<FormState>(emptyForm)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [isLoadingOptions, setIsLoadingOptions] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  useEffect(() => {
    let isMounted = true

    async function loadOptions(): Promise<void> {
      try {
        const response = await fetch(`${API_BASE_URL}/categorical-options`)
        if (!response.ok) {
          throw new Error('Unable to load dropdown options from the API.')
        }

        const payload = (await response.json()) as { options: CategoryOptions }
        if (!isMounted) {
          return
        }

        setOptions(payload.options)
        setFormState(createDefaultForm(payload.options))
        setErrorMessage('')
      } catch (error) {
        if (!isMounted) {
          return
        }

        setErrorMessage(
          error instanceof Error
            ? error.message
            : 'Unable to connect to the inference API.',
        )
      } finally {
        if (isMounted) {
          setIsLoadingOptions(false)
        }
      }
    }

    void loadOptions()

    return () => {
      isMounted = false
    }
  }, [])

  function updateField<Key extends keyof FormState>(key: Key, value: FormState[Key]): void {
    setFormState((currentState) => ({
      ...currentState,
      [key]: value,
    }))
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    setIsSubmitting(true)
    setErrorMessage('')

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formState,
          PATIENT_AGE: Number(formState.PATIENT_AGE),
          NUM_CONDITIONS: Number(formState.NUM_CONDITIONS),
        }),
      })

      const payload = (await response.json()) as
        | PredictionResponse
        | { detail?: string }
      if (!response.ok) {
        const detail = 'detail' in payload ? payload.detail : undefined
        throw new Error(detail ?? 'Prediction request failed.')
      }

      setResult(payload as PredictionResponse)
    } catch (error) {
      setResult(null)
      setErrorMessage(
        error instanceof Error ? error.message : 'Prediction request failed.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const probability = result?.predicted_probability ?? 0
  const confidenceClass =
    result === null
      ? 'idle'
      : !result.high_risk
        ? 'ineligible'
        : probability >= 0.7
          ? 'high'
          : probability >= 0.45
            ? 'medium'
            : 'low'

  return (
    <main className="shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">Disease X treatment inference</span>
          <h1>Estimate antiviral treatment likelihood for a high-risk patient.</h1>
          <p className="lede">
            This interface first checks whether the patient qualifies as high risk.
            If they do, it runs the trained model and returns the estimated probability
            of antiviral treatment.
          </p>
        </div>

        <div className={`result-card ${confidenceClass}`}>
          <div className="result-label">
            {result !== null && !result.high_risk ? 'Eligibility result' : 'Predicted likelihood'}
          </div>
          <div className="result-value">
            {result === null
              ? '--'
              : !result.high_risk
                ? 'Not high risk'
                : formatPercent(probability)}
          </div>
          <div className="result-status">
            {result === null
              ? 'Submit the form to validate risk and score a patient record.'
              : !result.high_risk
                ? result.message
                : result.prediction === 1
                ? 'Model prediction: likely treated'
                : 'Model prediction: less likely treated'}
          </div>
          <dl className="result-meta">
            <div>
              <dt>High risk</dt>
              <dd>
                {result === null ? 'Pending' : result.high_risk ? 'Yes' : 'No'}
              </dd>
            </div>
            <div>
              <dt>Population</dt>
              <dd>Age and condition rule</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{result?.model_filename ?? 'Awaiting eligibility check'}</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="workspace-card">
        <div className="workspace-header">
          <div>
            <h2>Patient inputs</h2>
            <p>
              High-risk eligibility is computed from the patient age and the selected
              underlying conditions before inference is attempted.
            </p>
          </div>
          <div className={`api-pill ${errorMessage ? 'offline' : 'online'}`}>
            {isLoadingOptions ? 'Loading API schema...' : errorMessage ? 'API issue' : 'API connected'}
          </div>
        </div>

        <form className="prediction-form" onSubmit={handleSubmit}>
          <label className="field field-number">
            <span>Patient age</span>
            <input
              min="0"
              type="number"
              value={formState.PATIENT_AGE}
              onChange={(event) => updateField('PATIENT_AGE', event.target.value)}
              required
            />
          </label>

          <label className="field field-number">
            <span>Number of conditions</span>
            <input
              min="0"
              type="number"
              value={formState.NUM_CONDITIONS}
              onChange={(event) => updateField('NUM_CONDITIONS', event.target.value)}
              required
            />
          </label>

          {categoryFields.map(({ key, label }) => (
            key === 'UNDERLYING_CONDITIONS' ? null :
            <label className="field" key={key}>
              <span>{label}</span>
              <select
                value={formState[key]}
                onChange={(event) => updateField(key, event.target.value)}
                disabled={isLoadingOptions || options === null}
                required
              >
                <option value="" disabled>
                  Select {label.toLowerCase()}
                </option>
                {(options?.[key] ?? []).map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          ))}

          <fieldset className="conditions-field">
            <legend>Underlying conditions used for high-risk validation</legend>
            <div className="conditions-grid">
              {(options?.UNDERLYING_CONDITIONS ?? []).map((condition) => {
                const isSelected = formState.UNDERLYING_CONDITIONS.includes(condition)
                return (
                  <label className={`condition-chip ${isSelected ? 'selected' : ''}`} key={condition}>
                    <input
                      checked={isSelected}
                      type="checkbox"
                      onChange={(event) => {
                        const nextConditions = event.target.checked
                          ? [...formState.UNDERLYING_CONDITIONS, condition]
                          : formState.UNDERLYING_CONDITIONS.filter(
                              (item) => item !== condition,
                            )
                        updateField('UNDERLYING_CONDITIONS', nextConditions)
                      }}
                    />
                    <span>{condition}</span>
                  </label>
                )
              })}
            </div>
          </fieldset>

          <div className="form-footer">
            <div className="form-note">
              This sends a JSON POST request to <span>{API_BASE_URL}/predict</span>.
            </div>
            <button type="submit" disabled={isSubmitting || isLoadingOptions || options === null}>
              {isSubmitting ? 'Scoring patient...' : 'Estimate likelihood'}
            </button>
          </div>
        </form>

        {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
      </section>
    </main>
  )
}

export default App
