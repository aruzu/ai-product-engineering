<?php

namespace App\Models;

class Persona implements \JsonSerializable
{
    public function __construct(
        public string $name,
        public string $context,
        public array $goals,
        public array $pains,
        public string $behavior
    ) {}

    public function jsonSerialize(): array
    {
        return [
            'name' => $this->name,
            'context' => $this->context,
            'goals' => $this->goals,
            'pains' => $this->pains,
            'behavior' => $this->behavior
        ];
    }

    public static function fromArray(array $data): self
    {
        return new self(
            $data['name'],
            $data['context'],
            $data['goals'],
            $data['pains'],
            $data['behavior']
        );
    }
} 