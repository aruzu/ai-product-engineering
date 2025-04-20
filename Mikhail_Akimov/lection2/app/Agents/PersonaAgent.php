<?php

namespace App\Agents;

use NeuronAI\Agent;
use NeuronAI\SystemPrompt;
use NeuronAI\Providers\AIProviderInterface;
use NeuronAI\Providers\OpenAI\OpenAI;
use App\Models\Feature;
use App\Models\Persona;
use App\Services\OpenAIService;

class PersonaAgent extends Agent
{
    public Persona $persona;

    public function saveParams(Persona $persona)
    {
        $this->persona = $persona;
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
                "Вы выступаете в роли {$this->persona->name}, {$this->persona->context}. " .
                "Ваши основные потребности: " . implode(', ', $this->persona->goals) . ". " .
                "Ваши основные проблемы: " . implode(', ', $this->persona->pains) . ". "
            ],
            steps: [
                "Отвечайте от первого лица, сохраняя характер и приоритеты персоны."
            ],
            output: [
                "Ведите диалог структурированно"
            ]            
        );
    }

    // public function getResponse(Feature $feature): string
    // {
    //     $systemMessage = "Вы выступаете в роли {$this->persona->name}, {$this->persona->description}. " .
    //         "Ваши основные потребности: {$this->persona->needs}. " .
    //         "Ваши основные проблемы: {$this->persona->painPoints}. " .
    //         "Отвечайте от первого лица, сохраняя характер и приоритеты персоны.";

    //     $response = $this->client->chat()->create([
    //         'model' => 'gpt-4.1-mini',
    //         'messages' => [
    //             ['role' => 'system', 'content' => $systemMessage],
    //             ['role' => 'user', 'content' => "Что вы думаете о следующей функции продукта?\n\n" .
    //                 "Название: {$feature->name}\n" .
    //                 "Описание: {$feature->description}\n" .
    //                 "Ценность: {$feature->value}\n\n" .
    //                 "Пожалуйста, поделитесь вашим мнением о полезности этой функции, " .
    //                 "насколько она решает ваши проблемы, и какие у вас есть предложения по её улучшению."]
    //         ],
    //         'temperature' => 0.7,
    //         'max_tokens' => 500
    //     ]);

    //     return $response->choices[0]->message->content;
    // }

    public function getPersona(): Persona
    {
        return $this->persona;
    }
} 