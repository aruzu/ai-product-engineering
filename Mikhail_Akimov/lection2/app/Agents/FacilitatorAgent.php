<?php

namespace App\Agents;

use NeuronAI\Agent;
use NeuronAI\SystemPrompt;
use NeuronAI\Chat\Messages\UserMessage;
use NeuronAI\Providers\AIProviderInterface;
use NeuronAI\Providers\OpenAI\OpenAI;
use App\Models\Feature;
use App\Models\Persona;
use App\Services\OpenAIService;


class FacilitatorAgent extends Agent
{
    private array $features;
    private array $personas;
    private array $personaAgents;
    private array $conversation = [];

    public function saveParams(array $features, array $personas, array $personaAgents)
    {
        $this->features = $features;
        $this->personas = $personas;
        $this->personaAgents = $personaAgents;
    }

    public function provider(): AIProviderInterface
    {
        return new OpenAI(
            key: OpenAIService::$apiKey,
            model: 'gpt-4.1-mini'
        );
    }

    public function instructions(): string
    {
        return new SystemPrompt(
            background: [
                "Вы - фасилитатор обсуждения новых фич продукта",
                "Ваша задача - провести продуктивное обсуждение с разными персонами"
            ],
            steps: [
                "Представьте каждую фичу",
                "Опросите каждую персону о понимании и использовании фичи",
                "Выявите опасения и ожидания",
                "В конце спросите о дополнительных пожеланиях"
            ],
            output: [
                "Ведите диалог структурированно",
                "Фиксируйте ключевые моменты обсуждения",
                "Подводите промежуточные итоги"
            ]
        );
    }

    public function conductDiscussion(): void
    {
        foreach ($this->features as $feature) {
            $this->discussFeature($feature);
        }

        $this->askFinalQuestions();
        $this->generateSummary();
    }

    private function discussFeature(Feature $feature): void
    {
        $this->logMessage("Фасилитатор", "Давайте обсудим фичу: {$feature->name}");
        $this->logMessage("Фасилитатор", "Описание: {$feature->description}");

        foreach ($this->personaAgents as $index => $agent) {
            $this->askPersonaAboutFeature($agent, $feature);
        }
    }

    private function askPersonaAboutFeature(PersonaAgent $agent, Feature $feature): void
    {
        $questions = [
            "Понимаете ли вы, зачем нужна эта фича?",
            "Как бы вы использовали эту фичу в своём контексте?",
            "Что вас волнует в этой фиче?"
        ];

        foreach ($questions as $question) {
            $this->logMessage("Фасилитатор", "{$agent->persona->name}, {$question}");

            $response = $agent->chat(new UserMessage($question));
            $this->logMessage($agent->persona->name, $response->getContent());
        }
    }

    private function askFinalQuestions(): void
    {
        foreach ($this->personaAgents as $agent) {
            $this->logMessage("Фасилитатор", "{$agent->persona->name}, хотите ли вы добавить что-то ещё?");
            
            $response = $agent->chat(
                new UserMessage("Хотите ли вы добавить что-то ещё к обсуждению фич?")
            );
            $this->logMessage($agent->persona->name, $response->getContent());
        }
    }

    private function generateSummary(): void
    {
        $prompt = "На основе проведенного обсуждения составьте итоговое резюме, включающее:\n" .
                "1. Что важно для пользователей\n" .
                "2. Какие моменты остались неясными\n" .
                "3. Какие фичи приоритетны\n\n" .
                "История обсуждения:\n" . 
                implode("\n", array_map(function($msg) {
                    return "[{$msg['speaker']}]: {$msg['message']}";
                }, $this->conversation));

        $response = $this->chat(new UserMessage($prompt));
        file_put_contents('summary.txt', $response->getContent());
    }

    private function logMessage(string $speaker, string $message): void
    {
        $this->conversation[] = [
            'speaker' => $speaker,
            'message' => $message,
            'timestamp' => date('Y-m-d H:i:s')
        ];
        
        $logEntry = "[{$speaker}]: {$message}\n";
        file_put_contents('conversation.txt', $logEntry, FILE_APPEND);
    }
} 