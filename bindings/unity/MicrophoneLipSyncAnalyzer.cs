using System;
using UnityEngine;
using UnityEngine.Events;

namespace TobyApi.LipSync
{
    [RequireComponent(typeof(AudioSource))]
    public sealed class MicrophoneLipSyncAnalyzer : MonoBehaviour
    {
        [Serializable] public sealed class VowelEvent : UnityEvent<LipSyncVowel> { }
        [Serializable] public sealed class ClassEvent : UnityEvent<LipSyncClass> { }

        [SerializeField] private AudioSource audioSource;
        [SerializeField, Range(256, 4096)] private int frameSampleCount = 1024;
        [SerializeField, Range(0.02f, 0.25f)] private float analysisIntervalSeconds = 0.05f;
        [SerializeField] private bool singingMode;
        [SerializeField] private bool enableTinyNn;
        [SerializeField] private bool enableGmm;
        [SerializeField] private bool robustLoudness = true;
        [SerializeField] private bool useAudioSourceTimeForTimedCues = true;
        [SerializeField, Range(0f, 1f)] private float metadataWeight = 0.55f;
        [SerializeField, Range(0f, 0.95f)] private float smoothing = 0.18f;
        [SerializeField, Range(0.005f, 0.5f)] private float loudnessAdaptation = 0.07f;
        [SerializeField] private bool logClassChanges;
        [SerializeField] private VowelEvent onVowelChanged = new VowelEvent();
        [SerializeField] private UnityEvent onVowelLost = new UnityEvent();
        [SerializeField] private ClassEvent onClassChanged = new ClassEvent();
        [SerializeField] private string currentClassName = "";
        [SerializeField] private string currentVowelName = "";
        [SerializeField] private float jawOpen;
        [SerializeField] private float vowelConfidence;
        [SerializeField] private float f1Hz;
        [SerializeField] private float f2Hz;

        private float[] interleavedBuffer;
        private float[] monoBuffer;
        private float nextAnalysisAt;
        private LipSyncAnalyzer analyzer;
        private int analyzerSampleRate;
        private LipSyncOptions analyzerOptions;
        private LipSyncTimedCue[] pendingTimedCues;
        private bool hasClass;

        public bool HasVowel { get; private set; }
        public LipSyncVowel CurrentVowel { get; private set; }
        public LipSyncClass CurrentClass { get; private set; }
        public LipSyncFrame CurrentFrame { get; private set; }
        public float JawOpen => jawOpen;
        public VowelEvent OnVowelChanged => onVowelChanged;
        public UnityEvent OnVowelLost => onVowelLost;
        public ClassEvent OnClassChanged => onClassChanged;

        private void Reset() { audioSource = GetComponent<AudioSource>(); }
        private void Awake() { if (audioSource == null) audioSource = GetComponent<AudioSource>(); }
        private void OnDestroy() { DisposeAnalyzer(); }

        private void Update()
        {
            if (Time.unscaledTime < nextAnalysisAt) return;
            nextAnalysisAt = Time.unscaledTime + analysisIntervalSeconds;
            TryAnalyzeNow(out _);
        }

        public void SetTimedCues(LipSyncTimedCue[] cues)
        {
            pendingTimedCues = cues;
            analyzer?.SetTimedCues(cues);
        }

        public void ClearTimedCues()
        {
            pendingTimedCues = null;
            analyzer?.ClearTimedCues();
        }

        public bool TryAnalyzeNow(out LipSyncFrame frame)
        {
            frame = default(LipSyncFrame);
            if (!TryReadCurrentFrame() || !EnsureAnalyzer(audioSource.clip.frequency))
            {
                ClearVowel();
                return false;
            }

            bool processed = useAudioSourceTimeForTimedCues
                ? analyzer.ProcessAtTime(monoBuffer, audioSource.time, out frame)
                : analyzer.Process(monoBuffer, out frame);

            if (!processed)
            {
                ClearVowel();
                return false;
            }

            CurrentFrame = frame;
            jawOpen = frame.jawOpen;
            vowelConfidence = frame.vowelConfidence;
            f1Hz = frame.f1Hz;
            f2Hz = frame.f2Hz;

            LipSyncClass bestClass = frame.BestClass;
            bool classChanged = !hasClass || CurrentClass != bestClass;
            hasClass = true;
            CurrentClass = bestClass;
            currentClassName = bestClass.ToString();

            if (classChanged)
            {
                if (logClassChanges) Debug.Log($"Lip sync class: {bestClass} jaw={jawOpen:0.00}", this);
                onClassChanged.Invoke(bestClass);
            }

            UpdateVowelState(bestClass);
            return true;
        }

        public bool TryRecognizeVowelNow(out LipSyncVowel vowel)
        {
            vowel = default(LipSyncVowel);
            if (!TryAnalyzeNow(out LipSyncFrame frame)) return false;
            return TryMapVowel(frame.BestClass, out vowel);
        }

        private bool EnsureAnalyzer(int sampleRate)
        {
            LipSyncOptions options = BuildOptions(sampleRate);
            if (analyzer != null && analyzer.IsValid && analyzerSampleRate == sampleRate && OptionsEqual(analyzerOptions, options)) return true;

            DisposeAnalyzer();
            if (!LipSync.TryCreateAnalyzer(options, out analyzer))
            {
                analyzer = null;
                analyzerSampleRate = 0;
                return false;
            }

            analyzerSampleRate = sampleRate;
            analyzerOptions = options;
            if (pendingTimedCues != null && pendingTimedCues.Length > 0) analyzer.SetTimedCues(pendingTimedCues);
            return true;
        }

        private LipSyncOptions BuildOptions(int sampleRate)
        {
            LipSyncOptions options = LipSyncOptions.Create(sampleRate, singingMode, enableTinyNn, pendingTimedCues != null && pendingTimedCues.Length > 0, robustLoudness, enableGmm);
            options.metadataWeight = metadataWeight;
            options.smoothing = singingMode ? Mathf.Max(smoothing, 0.55f) : smoothing;
            options.loudnessAdaptation = loudnessAdaptation;
            return options;
        }

        private static bool OptionsEqual(LipSyncOptions a, LipSyncOptions b)
        {
            return a.sampleRate == b.sampleRate && a.flags == b.flags
                && Mathf.Approximately(a.metadataWeight, b.metadataWeight)
                && Mathf.Approximately(a.smoothing, b.smoothing)
                && Mathf.Approximately(a.loudnessAdaptation, b.loudnessAdaptation);
        }

        private void DisposeAnalyzer()
        {
            if (analyzer == null) return;
            analyzer.Dispose();
            analyzer = null;
            analyzerSampleRate = 0;
        }

        private void UpdateVowelState(LipSyncClass bestClass)
        {
            if (!TryMapVowel(bestClass, out LipSyncVowel vowel))
            {
                ClearVowel();
                return;
            }

            bool changed = !HasVowel || CurrentVowel != vowel;
            HasVowel = true;
            CurrentVowel = vowel;
            currentVowelName = vowel.ToString();
            if (changed) onVowelChanged.Invoke(vowel);
        }

        private static bool TryMapVowel(LipSyncClass lipSyncClass, out LipSyncVowel vowel)
        {
            switch (lipSyncClass)
            {
                case LipSyncClass.A: vowel = LipSyncVowel.A; return true;
                case LipSyncClass.I: vowel = LipSyncVowel.I; return true;
                case LipSyncClass.U: vowel = LipSyncVowel.U; return true;
                case LipSyncClass.E: vowel = LipSyncVowel.E; return true;
                case LipSyncClass.O: vowel = LipSyncVowel.O; return true;
                default: vowel = default(LipSyncVowel); return false;
            }
        }

        private bool TryReadCurrentFrame()
        {
            if (audioSource == null || audioSource.clip == null || !audioSource.isPlaying) return false;
            AudioClip clip = audioSource.clip;
            int channels = Mathf.Max(1, clip.channels);
            int frameCount = Mathf.Clamp(frameSampleCount, 1, clip.samples);
            EnsureBuffers(frameCount, channels);

            int startFrame = audioSource.timeSamples - frameCount;
            while (startFrame < 0) startFrame += clip.samples;
            if (!clip.GetData(interleavedBuffer, startFrame)) return false;

            if (channels == 1)
            {
                Array.Copy(interleavedBuffer, monoBuffer, frameCount);
                return true;
            }

            for (int frame = 0; frame < frameCount; frame++)
            {
                float sum = 0f;
                int channelOffset = frame * channels;
                for (int channel = 0; channel < channels; channel++) sum += interleavedBuffer[channelOffset + channel];
                monoBuffer[frame] = sum / channels;
            }
            return true;
        }

        private void EnsureBuffers(int frameCount, int channels)
        {
            int interleavedLength = frameCount * channels;
            if (interleavedBuffer == null || interleavedBuffer.Length != interleavedLength) interleavedBuffer = new float[interleavedLength];
            if (monoBuffer == null || monoBuffer.Length != frameCount) monoBuffer = new float[frameCount];
        }

        private void ClearVowel()
        {
            if (!HasVowel) return;
            HasVowel = false;
            currentVowelName = "";
            onVowelLost.Invoke();
        }
    }
}
