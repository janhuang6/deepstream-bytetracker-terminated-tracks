//
// Originally created by Mayur Kulkarni on 11/11/21.
// Modified by Jan Huang on 5/14/25
//
#ifndef DNSTARPROD_TRACKER_H
#define DNSTARPROD_TRACKER_H

#include "nvdstracker.h"
#include "BYTETracker.h"
#include <memory>

#define MAX_TARGETS_PER_STREAM 512

/**
 * @brief Context for input video streams
 *
 * The stream context holds all necessary state to perform multi-object tracking
 * within the stream.
 */
class NvMOTContext
{
public:
    NvMOTContext(const NvMOTConfig &configIn, NvMOTConfigResponse& configResponse);

    ~NvMOTContext();

    /**
     * @brief Process a batch of frames
     *
     * Internal implementation of NvMOT_Process()
     *
     * @param [in] pParam Pointer to parameters for the frame to be processed
     * @param [out] pTrackedObjectsBatch Pointer to object tracks output
     */
    NvMOTStatus processFrame(const NvMOTProcessParams *params,
                                   NvMOTTrackedObjBatch *pTrackedObjectsBatch);
    /**
     * @brief Output the miscellaneous data if there are
     *
     *  Internal implementation of retrieveMiscData()
     *
     * @param [in] pParam Pointer to parameters for the frame to be processed
     * @param [out] pTrackerMiscData Pointer to miscellaneous data output
     */
    NvMOTStatus retrieveMiscData(const NvMOTProcessParams *params,
                                   NvMOTTrackerMiscData *pTrackerMiscData);
    /**
     * @brief Terminate trackers and release resources for a stream when the stream is removed
     *
     *  Internal implementation of NvMOT_RemoveStreams()
     *
     * @param [in] streamIdMask removed stream ID
     */
    NvMOTStatus removeStream(const NvMOTStreamId streamIdMask);

protected:

    // MODIFIED: Multi-stream/multi-class support via vector of ByteTrackers per stream
    std::map<uint64_t, std::vector<std::shared_ptr<BYTETracker>>> byteTrackerMap;

    // MODIFIED: Structure to hold terminated track info for DeepStream
    struct TerminatedTrackInfo {
        uint64_t streamID;
        uint64_t trackingId;
        int classId;
        float lastConfidence;
        NvMOTRect lastBbox;

        bool operator==(const TerminatedTrackInfo& other) const {
            return streamID == other.streamID &&
                   trackingId == other.trackingId &&
                   classId == other.classId &&
                   fabs(lastConfidence - other.lastConfidence) < 1e-5 &&
                   fabs(lastBbox.x - other.lastBbox.x) < 1e-5 &&
                   fabs(lastBbox.y - other.lastBbox.y) < 1e-5 &&
                   fabs(lastBbox.width - other.lastBbox.width) < 1e-5 &&
                   fabs(lastBbox.height - other.lastBbox.height) < 1e-5;
        }
    };
    std::vector<TerminatedTrackInfo> terminatedTracks;

    // MODIFIED: Loads tracker config (class modes, etc.)
    void loadTrackerConfig(const std::string& filePath);
};

#endif //DNSTARPROD_TRACKER_H