<?php

namespace App\Services;


use OpenAI\Client as OpenAIClient;
use OpenAI\Transporters\HttpTransporter;
use GuzzleHttp\Client as GuzzleClient;
use OpenAI\ValueObjects\ApiKey;
use OpenAI\ValueObjects\Transporter\BaseUri;
use OpenAI\ValueObjects\Transporter\Headers;
use OpenAI\ValueObjects\Transporter\QueryParams;
use OpenAI\ValueObjects\Transporter\Payload;
use OpenAI\Enums\Transporter\Method;

use App\Models\Feature;
use App\Models\Persona;

class OpenAIService
{
    private OpenAIClient $client;
    static public ?string $apiKey = null;

    public function __construct()
    {
        $httpClient = new GuzzleClient();
        $baseUri = BaseUri::from('https://api.openai.com/v1');
        $headers = Headers::withAuthorization(ApiKey::from(OpenAIService::$apiKey));
        $queryParams = QueryParams::create();
        
        $transporter = new HttpTransporter(
            $httpClient,
            $baseUri,
            $headers,
            $queryParams,
            function ($request) {
                return $request;
            }
        );
        
        $this->client = new OpenAIClient($transporter);
    }

    public function analyzeFeedback(string $feedback): array
    {
        $response = $this->client->chat()->create([
            'model' => 'gpt-4.1-mini',
            'messages' => [
                [
                    'role' => 'system',
                    'content' => 'Вы - аналитик пользовательских отзывов. Проанализируйте отзывы и предоставьте результат в следующем формате JSON:
                    {
                        "features": [
                            {
                                "name": "Название фичи",
                                "description": "Подробное описание фичи",
                                "userPains": ["Боль 1", "Боль 2"]
                            }
                        ],
                        "personas": [
                            {
                                "name": "Имя персоны",
                                "context": "Контекст использования продукта",
                                "goals": ["Цель 1", "Цель 2"],
                                "pains": ["Боль 1", "Боль 2"],
                                "behavior": "Описание поведения"
                            }
                        ]
                    }'
                ],
                [
                    'role' => 'user',
                    'content' => $feedback
                ]
            ],
            'temperature' => 0.7
        ]);

        $content = $response->choices[0]->message->content;
        return $this->parseAnalysisResponse($content);
    }

    private function parseAnalysisResponse(string $response): array
    {
        $data = json_decode($response, true);
        
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new \RuntimeException('Ошибка парсинга ответа от OpenAI: ' . json_last_error_msg());
        }

        $features = array_map(function ($featureData) {
            return Feature::fromArray($featureData);
        }, $data['features'] ?? []);

        $personas = array_map(function ($personaData) {
            return Persona::fromArray($personaData);
        }, $data['personas'] ?? []);

        return [
            'features' => $features,
            'personas' => $personas
        ];
    }
} 