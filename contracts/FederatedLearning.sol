// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FederatedLearning
 * @dev Smart contract for coordinating federated learning with zero-knowledge proofs
 * @author Flower-Homomorphic-Encryption Team
 */
contract FederatedLearning {
    
    // Events
    event ModelUpdateSubmitted(
        string indexed clientId,
        uint256 indexed round,
        string commitment,
        string zkProof,
        uint256 timestamp
    );
    
    event RoundCompleted(
        uint256 indexed round,
        uint256 totalUpdates,
        uint256 verifiedUpdates,
        string globalModelCommitment
    );
    
    event ClientRegistered(
        string indexed clientId,
        address clientAddress,
        uint256 timestamp
    );
    
    // Structs
    struct ModelUpdate {
        string clientId;
        uint256 round;
        string modelCommitment;
        string zkProof;
        uint256 timestamp;
        bool verified;
        address submitter;
    }
    
    struct Round {
        uint256 roundNumber;
        uint256 startTime;
        uint256 endTime;
        uint256 totalUpdates;
        uint256 verifiedUpdates;
        string globalModelCommitment;
        bool completed;
    }
    
    struct Client {
        string clientId;
        address clientAddress;
        uint256 registrationTime;
        uint256 totalUpdates;
        uint256 verifiedUpdates;
        bool active;
    }
    
    // State variables
    address public owner;
    uint256 public currentRound;
    uint256 public minClientsPerRound;
    uint256 public maxRoundDuration; // in seconds
    
    // Mappings
    mapping(uint256 => mapping(string => ModelUpdate)) public roundUpdates;
    mapping(uint256 => string[]) public roundClientIds;
    mapping(uint256 => Round) public rounds;
    mapping(string => Client) public clients;
    mapping(address => string) public addressToClientId;
    
    // Arrays for iteration
    string[] public registeredClientIds;
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyRegisteredClient() {
        require(bytes(addressToClientId[msg.sender]).length > 0, "Client not registered");
        _;
    }
    
    modifier validClientId(string memory clientId) {
        require(bytes(clientId).length > 0, "Client ID cannot be empty");
        _;
    }
    
    modifier validCommitment(string memory commitment) {
        require(bytes(commitment).length > 0, "Model commitment cannot be empty");
        _;
    }
    
    // Constructor
    constructor(uint256 _minClientsPerRound, uint256 _maxRoundDuration) {
        owner = msg.sender;
        currentRound = 1;
        minClientsPerRound = _minClientsPerRound;
        maxRoundDuration = _maxRoundDuration;
        
        // Initialize first round
        rounds[currentRound] = Round({
            roundNumber: currentRound,
            startTime: block.timestamp,
            endTime: 0,
            totalUpdates: 0,
            verifiedUpdates: 0,
            globalModelCommitment: "",
            completed: false
        });
    }
    
    /**
     * @dev Register a new client for federated learning
     * @param clientId Unique identifier for the client
     */
    function registerClient(string memory clientId) 
        external 
        validClientId(clientId) 
    {
        require(clients[clientId].clientAddress == address(0), "Client already registered");
        require(bytes(addressToClientId[msg.sender]).length == 0, "Address already registered");
        
        clients[clientId] = Client({
            clientId: clientId,
            clientAddress: msg.sender,
            registrationTime: block.timestamp,
            totalUpdates: 0,
            verifiedUpdates: 0,
            active: true
        });
        
        addressToClientId[msg.sender] = clientId;
        registeredClientIds.push(clientId);
        
        emit ClientRegistered(clientId, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Submit a model update with zero-knowledge proof
     * @param clientId Client identifier
     * @param round Round number
     * @param modelCommitment Cryptographic commitment to model parameters
     * @param zkProof Zero-knowledge proof of correct training
     */
    function submitModelUpdate(
        string memory clientId,
        uint256 round,
        string memory modelCommitment,
        string memory zkProof
    ) 
        external 
        onlyRegisteredClient
        validClientId(clientId)
        validCommitment(modelCommitment)
    {
        require(keccak256(bytes(clientId)) == keccak256(bytes(addressToClientId[msg.sender])), 
                "Client ID mismatch");
        require(round == currentRound, "Invalid round number");
        require(roundUpdates[round][clientId].timestamp == 0, "Update already submitted for this round");
        require(!rounds[round].completed, "Round already completed");
        
        // Verify round hasn't exceeded time limit
        if (maxRoundDuration > 0) {
            require(block.timestamp <= rounds[round].startTime + maxRoundDuration, 
                    "Round time limit exceeded");
        }
        
        // Simple ZK proof verification (in practice, this would be more sophisticated)
        bool proofValid = _verifyZKProof(zkProof, modelCommitment, clientId);
        
        // Store the update
        roundUpdates[round][clientId] = ModelUpdate({
            clientId: clientId,
            round: round,
            modelCommitment: modelCommitment,
            zkProof: zkProof,
            timestamp: block.timestamp,
            verified: proofValid,
            submitter: msg.sender
        });
        
        roundClientIds[round].push(clientId);
        rounds[round].totalUpdates++;
        
        if (proofValid) {
            rounds[round].verifiedUpdates++;
            clients[clientId].verifiedUpdates++;
        }
        
        clients[clientId].totalUpdates++;
        
        emit ModelUpdateSubmitted(clientId, round, modelCommitment, zkProof, block.timestamp);
        
        // Check if round should be completed
        _checkRoundCompletion(round);
    }
    
    /**
     * @dev Complete the current round and start a new one
     * @param globalModelCommitment Commitment to the aggregated global model
     */
    function completeRound(string memory globalModelCommitment) 
        external 
        onlyOwner 
    {
        Round storage round = rounds[currentRound];
        require(!round.completed, "Round already completed");
        require(round.verifiedUpdates >= minClientsPerRound, "Insufficient verified updates");
        
        round.endTime = block.timestamp;
        round.globalModelCommitment = globalModelCommitment;
        round.completed = true;
        
        emit RoundCompleted(
            currentRound, 
            round.totalUpdates, 
            round.verifiedUpdates, 
            globalModelCommitment
        );
        
        // Start new round
        currentRound++;
        rounds[currentRound] = Round({
            roundNumber: currentRound,
            startTime: block.timestamp,
            endTime: 0,
            totalUpdates: 0,
            verifiedUpdates: 0,
            globalModelCommitment: "",
            completed: false
        });
    }
    
    /**
     * @dev Get all model updates for a specific round
     * @param round Round number
     * @return clientIds Array of client IDs that submitted updates
     * @return updates Array of model updates
     */
    function getRoundUpdates(uint256 round) 
        external 
        view 
        returns (string[] memory clientIds, ModelUpdate[] memory updates) 
    {
        string[] memory roundClients = roundClientIds[round];
        ModelUpdate[] memory roundModelUpdates = new ModelUpdate[](roundClients.length);
        
        for (uint256 i = 0; i < roundClients.length; i++) {
            roundModelUpdates[i] = roundUpdates[round][roundClients[i]];
        }
        
        return (roundClients, roundModelUpdates);
    }
    
    /**
     * @dev Get information about a specific round
     * @param round Round number
     * @return Round information
     */
    function getRoundInfo(uint256 round) 
        external 
        view 
        returns (Round memory) 
    {
        return rounds[round];
    }
    
    /**
     * @dev Get client information
     * @param clientId Client identifier
     * @return Client information
     */
    function getClientInfo(string memory clientId) 
        external 
        view 
        returns (Client memory) 
    {
        return clients[clientId];
    }
    
    /**
     * @dev Get all registered client IDs
     * @return Array of all registered client IDs
     */
    function getAllClients() 
        external 
        view 
        returns (string[] memory) 
    {
        return registeredClientIds;
    }
    
    /**
     * @dev Get statistics for the current round
     * @return currentRoundNumber Current round number
     * @return totalUpdates Total updates submitted in current round
     * @return verifiedUpdates Verified updates in current round
     * @return timeRemaining Time remaining in current round (0 if no limit)
     */
    function getCurrentRoundStats() 
        external 
        view 
        returns (
            uint256 currentRoundNumber, 
            uint256 totalUpdates, 
            uint256 verifiedUpdates, 
            uint256 timeRemaining
        ) 
    {
        Round memory round = rounds[currentRound];
        uint256 remaining = 0;
        
        if (maxRoundDuration > 0 && !round.completed) {
            uint256 elapsed = block.timestamp - round.startTime;
            if (elapsed < maxRoundDuration) {
                remaining = maxRoundDuration - elapsed;
            }
        }
        
        return (currentRound, round.totalUpdates, round.verifiedUpdates, remaining);
    }
    
    /**
     * @dev Update contract configuration (only owner)
     * @param _minClientsPerRound Minimum clients required per round
     * @param _maxRoundDuration Maximum duration for each round
     */
    function updateConfiguration(uint256 _minClientsPerRound, uint256 _maxRoundDuration) 
        external 
        onlyOwner 
    {
        minClientsPerRound = _minClientsPerRound;
        maxRoundDuration = _maxRoundDuration;
    }
    
    /**
     * @dev Emergency function to force complete a round (only owner)
     */
    function forceCompleteRound() 
        external 
        onlyOwner 
    {
        Round storage round = rounds[currentRound];
        require(!round.completed, "Round already completed");
        
        round.endTime = block.timestamp;
        round.completed = true;
        
        emit RoundCompleted(
            currentRound, 
            round.totalUpdates, 
            round.verifiedUpdates, 
            "FORCE_COMPLETED"
        );
        
        // Start new round
        currentRound++;
        rounds[currentRound] = Round({
            roundNumber: currentRound,
            startTime: block.timestamp,
            endTime: 0,
            totalUpdates: 0,
            verifiedUpdates: 0,
            globalModelCommitment: "",
            completed: false
        });
    }
    
    // Internal functions
    
    /**
     * @dev Simple ZK proof verification (mock implementation)
     * In a real implementation, this would verify the actual ZK proof
     * @param zkProof Zero-knowledge proof string
     * @param commitment Model commitment
     * @param clientId Client identifier
     * @return Whether the proof is valid
     */
    function _verifyZKProof(
        string memory zkProof, 
        string memory commitment, 
        string memory clientId
    ) 
        internal 
        pure 
        returns (bool) 
    {
        // Mock verification - in practice, this would use a ZK verification library
        // Check that proof is not empty and follows expected format
        bytes memory proofBytes = bytes(zkProof);
        bytes memory commitmentBytes = bytes(commitment);
        bytes memory clientBytes = bytes(clientId);
        
        return (proofBytes.length > 0 && 
                commitmentBytes.length > 0 && 
                clientBytes.length > 0 &&
                proofBytes.length >= 32); // Minimum expected proof length
    }
    
    /**
     * @dev Check if round should be automatically completed
     * @param round Round number to check
     */
    function _checkRoundCompletion(uint256 round) 
        internal 
    {
        Round storage roundData = rounds[round];
        
        // Auto-complete if we have enough verified updates and all registered clients have submitted
        if (roundData.verifiedUpdates >= minClientsPerRound && 
            roundData.totalUpdates >= registeredClientIds.length) {
            
            // This could trigger automatic round completion
            // For now, we just emit an event to notify off-chain components
        }
    }
}