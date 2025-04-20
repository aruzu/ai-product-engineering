<?php

require_once __DIR__ . '/vendor/autoload.php';

use App\Services\OpenAIService;
use App\Agents\PersonaAgent;
use App\Agents\FacilitatorAgent;

use App\Models\Feature;
use App\Models\Persona;
use App\Models\Product;

// Загрузка переменных окружения
$dotenv = \Dotenv\Dotenv::createImmutable(__DIR__ . '/', '.env');
$dotenv->safeLoad();
$dotenv->required(['OPENAI_API_KEY'])->notEmpty();

// Проверка наличия API ключа
if (!$_ENV['OPENAI_API_KEY']) {
    die("Ошибка: Не установлен OPENAI_API_KEY в файле .env\n");
}

try {
    // Чтение файла с отзывами
    if (!file_exists('reports.txt')) {
        die("Ошибка: Файл reports.txt не найден\n");
    }

    OpenAIService::$apiKey = $_ENV['OPENAI_API_KEY'];

    if (! file_exists('features.json') || ! file_exists('personas.json')) {
        echo "Анализируем отзывы: reports.txt" . PHP_EOL;
        $feedback = file_get_contents('reports.txt');

        // Анализ отзывов через OpenAI
        $openAiService = new OpenAIService();
        $analysis = $openAiService->analyzeFeedback($feedback);

        $features = $analysis['features'];
        $personas = $analysis['personas'];

        // Сохранение результатов анализа
        file_put_contents('features.json', json_encode($features, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
        file_put_contents('personas.json', json_encode($personas, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
    } else {
        echo "Считываем данные из файлов: features.json и personas.json" . PHP_EOL;
        $features = json_decode(file_get_contents('features.json'), true);
        $personas = json_decode(file_get_contents('personas.json'), true);

        $features = array_map(function ($featureData) {
            return Feature::fromArray($featureData);
        }, $features ?? []);

        $personas = array_map(function ($personaData) {
            return Persona::fromArray($personaData);
        }, $personas ?? []);

    }

    echo "Анализ отзывов завершен. Найдено:\n";
    echo "- " . count($features) . " фич\n";
    echo "- " . count($personas) . " персон\n\n";

    // Создание агентов для каждой персоны
    $personaAgents = [];
    $i=0;
    foreach ($personas as $idx => $persona) {
        if ($i >= 3) break;
        // $personas[$idx] = $persona = Persona::fromArray($persona);
        $agent = PersonaAgent::make();
        $agent->saveParams($persona);
        $personaAgents[] = $agent;
        $i++;
    }

    // Создание и запуск агента-фасилитатора
    $facilitator = FacilitatorAgent::make();
    $facilitator->saveParams($features, $personas, $personaAgents);
    $facilitator->conductDiscussion();

    echo "Обсуждение завершено. Результаты сохранены в файлах:\n";
    echo "- conversation.txt - лог обсуждения\n";
    echo "- summary.txt - итоговое резюме\n";

} catch (\Exception $e) {
    die("Ошибка: " . $e->getMessage() . "\n");
} 