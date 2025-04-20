<?php

namespace App\Models;

class Product
{
    /** @var Feature[] */
    public array $features = [];

    public function addFeature(Feature $feature): void
    {
        $this->features[] = $feature;
    }

    public function getFeatures(): array
    {
        return $this->features;
    }

    public function getFeatureByName(string $name): ?Feature
    {
        foreach ($this->features as $feature) {
            if ($feature->name === $name) {
                return $feature;
            }
        }
        return null;
    }

    public function toArray(): array
    {
        return [
            'features' => array_map(fn(Feature $feature) => $feature->toArray(), $this->features)
        ];
    }

    public static function fromArray(array $data): self
    {
        $product = new self();
        foreach ($data['features'] ?? [] as $featureData) {
            $product->addFeature(Feature::fromArray($featureData));
        }
        return $product;
    }
} 