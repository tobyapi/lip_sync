using System;
using UnityEngine;
using UnityEngine.Events;

namespace TobyApi.LipSync
{
    [RequireComponent(typeof(AudioSource))]
    public sealed class MicrophoneLipSyncAnalyzer : MonoBehaviour
    {
        [Serializable] public sealed class ClassEvent : UnityEvent<LipSyncClass> { }

        [SerializeField] private AudioSource audioSource;
        [SerializeField, Range(256, 4096)] private int frameSampleCount = 1024;
        [SerializeField, Range(0.02f, 0.25f)] private float analysisIntervalSeconds = 0.05f;
        [SerializeField] private bool singingMode;
        [SerializeField] private bool enableTinyNn;
        [SerializeField] private bool enableGmm;
        [SerializeField] private bool robustLoudness = true;
        [SerializeField] private bool useAudioSourceTimeForTimedCues = true;
        [SerializeField] private LipSyncMapperKind mapperKind = LipSyncMapperKind.Vrm;
        [SerializeField, Range(0f, 1f)] private float metadataWeight = 0.55f;
        [SerializeField, Range(0f, 0.95f)] private float smoothing = 0.18f;
        [SerializeField, Range(0.005f, 0.5f)] private float loudnessAdaptation = 0.07f;
        [SerializeField] private bool logClassChanges;
        [SerializeField] private ClassEvent onClassChanged = new ClassEvent();
        [SerializeField] private string currentClassName = "";
        [SerializeField] private float jawOpen;
        [SerializeField] private float vowelConfidence;
        [SerializeField] private float f1Hz;
        [SerializeField] private float f2Hz;
        [SerializeField] private float aa;
        [SerializeField] private float ih;
        [SerializeField] private float ou;
        [SerializeField] private float ee;
        [SerializeField] private float oh;
        [SerializeField] private float mouthClose;
        [SerializeField] private float mouthFunnel;
        [SerializeField] private float mouthPucker;
        [SerializeField] private float mouthWide;

        private float[] interleavedBuffer;
        private float nextAnalysisAt;
        private LipSyncAnalyzer analyzer;
        private int analyzerSampleRate;
        private LipSyncOptions analyzerOptions;
        private LipSyncTimedCue[] pendingTimedCues;
        private bool hasClass;
        private bool hasMappedFrame;

        public LipSyncClass CurrentClass { get; private set; }
        public LipSyncFrame CurrentFrame { get; private set; }
        public LipSyncMappedFrame CurrentMappedFrame { get; private set; }
        public bool HasMappedFrame => hasMappedFrame;
        public float JawOpen => jawOpen;
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
            int frameCount;
            int channels;
            if (!TryReadCurrentFrame(out frameCount, out channels) || !EnsureAnalyzer(audioSource.clip.frequency))
            {
                ClearAnalysisState();
                return false;
            }

            bool processed = useAudioSourceTimeForTimedCues
                ? analyzer.ProcessInterleavedAtTime(interleavedBuffer, frameCount, channels, audioSource.time, out frame)
                : analyzer.ProcessInterleaved(interleavedBuffer, frameCount, channels, out frame);

            if (!processed)
            {
                ClearAnalysisState();
                return false;
            }

            CurrentFrame = frame;
            jawOpen = frame.jawOpen;
            vowelConfidence = frame.vowelConfidence;
            f1Hz = frame.f1Hz;
            f2Hz = frame.f2Hz;

            LipSyncMappedFrame mappedFrame;
            hasMappedFrame = LipSync.TryMapFrame(frame, mapperKind, out mappedFrame);
            CurrentMappedFrame = hasMappedFrame ? mappedFrame : LipSyncMappedFrame.Empty;
            UpdateMappedFields(mappedFrame, hasMappedFrame);

            LipSyncClass bestClass = hasMappedFrame ? mappedFrame.BestClass : frame.BestClass;
            bool classChanged = !hasClass || CurrentClass != bestClass;
            hasClass = true;
            CurrentClass = bestClass;
            currentClassName = bestClass.ToString();

            if (classChanged)
            {
                if (logClassChanges) Debug.Log($"Lip sync class: {bestClass} jaw={jawOpen:0.00}", this);
                onClassChanged.Invoke(bestClass);
            }

            return true;
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

        private bool TryReadCurrentFrame(out int frameCount, out int channels)
        {
            frameCount = 0;
            channels = 0;
            if (audioSource == null || audioSource.clip == null || !audioSource.isPlaying) return false;

            AudioClip clip = audioSource.clip;
            channels = Mathf.Max(1, clip.channels);
            frameCount = Mathf.Clamp(frameSampleCount, 1, clip.samples);
            EnsureBuffer(frameCount, channels);

            int startFrame = audioSource.timeSamples - frameCount;
            while (startFrame < 0) startFrame += clip.samples;
            return clip.GetData(interleavedBuffer, startFrame);
        }

        private void EnsureBuffer(int frameCount, int channels)
        {
            int interleavedLength = frameCount * channels;
            if (interleavedBuffer == null || interleavedBuffer.Length != interleavedLength) interleavedBuffer = new float[interleavedLength];
        }

        private void UpdateMappedFields(LipSyncMappedFrame mappedFrame, bool mapped)
        {
            if (!mapped)
            {
                aa = ih = ou = ee = oh = 0f;
                mouthClose = mouthFunnel = mouthPucker = mouthWide = 0f;
                return;
            }

            aa = mappedFrame.aa;
            ih = mappedFrame.ih;
            ou = mappedFrame.ou;
            ee = mappedFrame.ee;
            oh = mappedFrame.oh;
            mouthClose = mappedFrame.mouthClose;
            mouthFunnel = mappedFrame.mouthFunnel;
            mouthPucker = mappedFrame.mouthPucker;
            mouthWide = mappedFrame.mouthWide;
        }

        private void ClearAnalysisState()
        {
            hasClass = false;
            hasMappedFrame = false;
            CurrentClass = LipSyncClass.Rest;
            CurrentMappedFrame = LipSyncMappedFrame.Empty;
            currentClassName = "";
            aa = ih = ou = ee = oh = 0f;
            mouthClose = mouthFunnel = mouthPucker = mouthWide = 0f;
        }
    }
}