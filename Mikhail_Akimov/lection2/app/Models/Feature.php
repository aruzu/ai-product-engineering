<?php

namespace App\Models;

class Feature
{
    public string $name;
    public string $description;
    public string $value;

    public function __construct(string $name, string $description, string $value)
    {
        $this->name = $name;
        $this->description = $description;
        $this->value = $value;
    }

    public static function fromArray(array $data): self
    {
        return new self(
            $data['name'] ?? '',
            $data['description'] ?? '',
            $data['value'] ?? ''
        );
    }

    public function toArray(): array
    {
        return [
            'name' => $this->name,
            'description' => $this->description,
            'value' => $this->value
        ];
    }
} 